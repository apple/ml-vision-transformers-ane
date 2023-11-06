#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""
Reference: 
    [1] MOAT: https://arxiv.org/pdf/2210.01820.pdf.
    [2] Tensorflow official impl: https://github.com/google-research/deeplab2/blob/main/model/pixel_encoder/moat.py
"""
import logging
import math
from typing import Optional, Sequence, Any, Tuple

import collections
from vision_transformers.attention_utils import (
    WindowAttention,
    PEType,
    window_partition,
    window_reverse,
)
from vision_transformers.mbconv import MBConvBlock
import torch
from torch import nn
from torch.nn import GELU
from dataclasses import dataclass

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)

# Change namedtuple defaults
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


@dataclass
class MOATConfig:
    """MOAT config. Default values are from tiny_moat_0.

    For detailed model hyperparameter configuration, refer to the MOAT paper.
    Supports two attention modes of operation, which you may experiment with for tradeoff between PnP and KPI:
    1. For local attention, window size is limited to a fixed window size (up to user configuration depends on input resolution).
    2. For global attention, do attention on the full feature map.

    :param stem_size: hidden size of the conv stem.
    :param block_type: type of each stage.
    :param num_blocks: number of blocks for each stage.
    :param hidden_size: hidden size of each stage.
    :param window_size: window_size of each stage if using attention block.
    :param activation: activation function to use.
    :param attention_mode: use global or local attention.
    :param split_head: whether do split head attention. split_head makes it run more efficiently on ANE.

    """

    stem_size: Sequence[int] = (32, 32)
    block_type: Sequence[str] = ("mbconv", "mbconv", "moat", "moat")
    num_blocks: Sequence[int] = (2, 3, 7, 2)
    hidden_size: Sequence[int] = (32, 64, 128, 256)
    window_size: Sequence[Any] = (None, None, (14, 14), (7, 7))
    activation: nn.Module = GELU()
    attention_mode: str = "global"
    split_head: bool = True
    stage_stride: Sequence[int] = (2, 2, 2, 2)
    mbconv_block_expand_ratio: int = 4
    moat_block_expand_ratio: int = 4
    pe_type: PEType = PEType.LePE_ADD

    def __post_init__(self):
        if self.attention_mode == "local":
            # window_size should be limited to local context
            local_context_lower, local_context_upper = 6, 16
            for window in self.window_size:
                if window is not None:
                    assert isinstance(window, tuple) or isinstance(window, list)
                    for hw in window:
                        assert hw >= local_context_lower and hw <= local_context_upper


@dataclass
class MOATBlockConfig:
    """MOAT block config.

    :param block_name: name of the block.
    :param window_size: attention window size.
    :param attn_norm_class: normalization layer in attention.
    :param activation: activation function.
    :param head_dim: dimension of each head.
    :param kernel_size: kernel size for MBConv block
    :param stride: stride for MBConv block
    :param expand_ratio: expansion ratio in the MBConv block.
    :param id_skip: do skip connection or not.
    :param se_ratio: channel reduction ratio in squeeze and excitation, if 0 or None, no SE.
    :param attention_mode: use global or local attention.
    :param split_head: whether do split head attention. split_head makes it run more efficiently on ANE.
    :param pe_type: position embedding type

    """

    block_name: str = "moat_block"
    window_size: Optional[Sequence[int]] = None
    attn_norm_class: nn.Module = nn.LayerNorm
    head_dim: int = 32  # dim of each head
    activation: nn.Module = GELU()
    # BlockArgs
    kernel_size: int = 3
    stride: int = 1
    input_filters: int = 32
    output_filters: int = 32
    expand_ratio: int = 4
    id_skip: bool = True
    se_ratio: Optional[float] = None
    attention_mode: str = "global"
    split_head: bool = False
    pe_type: PEType = PEType.LePE_ADD


class Stem(nn.Sequential):
    """Convolutional stem consists of 2 convolution layers.

    :param dims: specifies the dimensions for the convolution stems.

    """

    def __init__(self, dims: Sequence[int]):
        stem_layers = []

        for i in range(len(dims)):
            norm_layer = None
            activation_layer = None

            if i == 0:
                activation_layer = GELU()
                norm_layer = True

            stride = 2 if i == 0 else 1
            in_channels = dims[i - 1] if i >= 1 else 3
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=dims[i],
                kernel_size=3,
                bias=True,
                stride=stride,
                padding="same"
                if stride == 1
                else 1,  # strided conv does not support "same"
            )
            stem_layers.append(conv_layer)
            if activation_layer is not None:
                stem_layers.append(activation_layer)
            if norm_layer:
                stem_layers.append(nn.BatchNorm2d(dims[i]))

        super().__init__(*stem_layers)


class MOATBlock(nn.Module):
    """A MOAT block consists of MBConv (w/o squeeze-excitation blocks) and MHSA.

    :param config: a MOATBlockConfig object to specify block dims, attention mode, window size, etc.

    """

    def __init__(self, config: MOATBlockConfig):
        super().__init__()
        block_args = BlockArgs(
            kernel_size=config.kernel_size,
            stride=config.stride,
            se_ratio=None,  # MOAT block does not use SE branch
            input_filters=config.input_filters,
            output_filters=config.output_filters,
            id_skip=True,
            expand_ratio=config.expand_ratio,
        )
        self._mbconv = MBConvBlock(
            block_args,
            activation="gelu",
            pre_norm=True,
        )

        # dim after MBConv block
        dim = config.output_filters

        # currently LN apply normalization to the last few dimensions, therefore need NHWC format.
        # see pytorch issue 71456: https://github.com/pytorch/pytorch/issues/71465
        self._attn_norm = config.attn_norm_class(
            normalized_shape=dim,
            eps=1e-5,
            elementwise_affine=True,
        )
        assert (
            dim % config.head_dim == 0
        ), "tensor dimension: {} can not divide by head_dim: {}.".format(
            dim, config.head_dim
        )
        num_heads = dim // config.head_dim
        print("######pe_type in MOATBlock: ", config.pe_type)
        self._window_attention = WindowAttention(
            dim,
            window_size=config.window_size,
            num_heads=num_heads,
            split_head=config.split_head,
            pe_type=config.pe_type,
        )
        self.window_size = config.window_size
        self.attention_mode = config.attention_mode

    def forward(self, inputs):
        """inputs: (batch_size, C, H, W)
        output: ((batch_size, C, H//stride, W//stride)

        :param inputs:

        """

        # MBConv block may contain downsampling layer
        output = self._mbconv(inputs)
        N, C, H, W = output.shape

        # shortcut is before LN in MOAT
        shortcut = output
        # transpose to prepare the tensor for window_partition
        output = output.permute((0, 2, 3, 1))  # NHWC

        assert (
            output.shape[-1] % 32 == 0
        ), "ANE buffer not aligned, last dim={}.".format(output.shape[-1])
        output = self._attn_norm(output)

        if self.attention_mode == "local":
            x_windows = window_partition(output, self.window_size)
            x_windows = x_windows.reshape(
                (-1, self.window_size[0] * self.window_size[1], C)
            )
            attn_windows = self._window_attention(x_windows)
            output = window_reverse(attn_windows, self.window_size, H, W)
        # No need for window_partion/reverse on low res input
        elif self.attention_mode == "global":
            global_attention_windows = output.reshape((N, H * W, C))
            output = self._window_attention(global_attention_windows)

        output = output.reshape((N, H, W, C)).permute((0, 3, 1, 2))  # NCHW

        # may add drop_path here for output
        output = shortcut + output
        return output  # NCHW


class MOAT(nn.Module):
    """MOAT model definition.

    :param config: a MOATConfig object to specify MOAT variant, attention mode, etc.

    """

    def __init__(self, config: MOATConfig):
        super().__init__()
        self._stem = Stem(dims=config.stem_size)
        # Need to use ModuleList (instead of vanilla python list) for the module to be properly registered
        self._blocks = nn.ModuleList()
        self.config = config
        for stage_id in range(len(config.block_type)):
            stage_blocks = nn.ModuleList()
            stage_input_filters = (
                config.hidden_size[stage_id - 1]
                if stage_id > 0
                else config.stem_size[-1]
            )
            stage_output_filters = config.hidden_size[stage_id]

            for local_block_id in range(config.num_blocks[stage_id]):
                block_stride = 1
                block_name = "block_{:0>2d}_{:0>2d}_".format(stage_id, local_block_id)

                if local_block_id == 0:  # downsample in the first block of each stage
                    block_stride = config.stage_stride[stage_id]
                    block_input_filters = stage_input_filters
                else:
                    block_input_filters = stage_output_filters

                if config.block_type[stage_id] == "mbconv":
                    block_args = BlockArgs(
                        kernel_size=3,
                        stride=block_stride,
                        se_ratio=0.25,  # SE block reduction ratio
                        input_filters=block_input_filters,
                        output_filters=stage_output_filters,
                        expand_ratio=config.mbconv_block_expand_ratio,
                        id_skip=True,
                    )
                    block = MBConvBlock(
                        block_args,
                        activation="gelu",
                        pre_norm=True,
                    )
                elif config.block_type[stage_id] == "moat":
                    print("######pe_type: ", config.pe_type)
                    block_config = MOATBlockConfig(
                        block_name=block_name,
                        stride=block_stride,
                        window_size=config.window_size[stage_id],
                        input_filters=block_input_filters,
                        output_filters=stage_output_filters,
                        attention_mode=config.attention_mode,
                        split_head=config.split_head,
                        expand_ratio=config.moat_block_expand_ratio,
                        pe_type=config.pe_type,
                    )
                    block = MOATBlock(block_config)
                else:
                    raise ValueError(
                        "Network type {} not defined.".format(config.block_type)
                    )

                stage_blocks.append(block)

            self._blocks.append(stage_blocks)

    def forward(self, inputs: torch.Tensor, out_indices: Sequence[int] = (0, 1, 2, 3)):
        """

        :param inputs: torch.Tensor:
        :param out_indices: Sequence[int]:  (Default value = (0)
        :param 1: param 2:
        :param 3:
        :param inputs: torch.Tensor:
        :param out_indices: Sequence[int]:  (Default value = (0)
        :param 2:
        :param 3):

        """
        outs = []
        output = self._stem(inputs)

        for stage_id, stage_blocks in enumerate(self._blocks):
            for block in stage_blocks:
                output = block(output)
            if stage_id in out_indices:
                outs.append(output)
        return outs


def get_stage_strides(output_stride):
    """

    :param output_stride:

    """
    if output_stride == 32:
        stage_stride = (2, 2, 2, 2)
    elif output_stride == 16:
        stage_stride = (2, 2, 2, 1)
    elif output_stride == 8:
        stage_stride = (2, 2, 1, 1)
    return stage_stride


def _build_model(
    shape: Sequence[int] = (1, 3, 192, 256),
    base_arch: str = "tiny-moat-2",
    attention_mode: str = "global",
    split_head: bool = True,
    output_stride: int = 32,
    channel_buffer_align: bool = True,
    num_blocks: Sequence[int] = (2, 3, 7, 2),
    mbconv_block_expand_ratio: int = 4,
    moat_block_expand_ratio: int = 4,
    local_window_size: Optional[Sequence[int]] = None,
    pe_type: PEType = PEType.LePE_ADD,
) -> Tuple[MOATConfig, MOAT]:
    """Construct MOAT models.

    :param shape: input shape to the model.
    :param base_arch: architecture variant of MOAT.
    :param attention_mode: global or local (window based) attention
    :param output_stride: stride of output with respect to the input res, e.g., 32 meaning output will be 1/32 of input res
    :param split_head: whether do split_head attention. split_head is enabled by default as it is faster on ANE.
    :param channel_buffer_align: if True, make channel divisible by 32
    :param num_blocks: number of blocks in each stage.
    :param mbconv_block_expand_ratio: expansion ratio of mbconv blocks in first 2 stages
    :param moat_block_expand_ratio: expansion ratio of moat blocks in last 2 stages
    :param local_window_size: local window size of attention.
    :param pe_type: position embedding type
    :param shape: Sequence[int]:  (Default value = (1)
    :param 3: param 192:
    :param 256: param base_arch: str:  (Default value = "tiny-moat-2")
    :param attention_mode: str:  (Default value = "global")
    :param split_head: bool:  (Default value = True)
    :param output_stride: int:  (Default value = 32)
    :param channel_buffer_align: bool:  (Default value = True)
    :param num_blocks: Sequence[int]:  (Default value = (2)
    :param 7: param 2):
    :param mbconv_block_expand_ratio: int:  (Default value = 4)
    :param moat_block_expand_ratio: int:  (Default value = 4)
    :param local_window_size: Optional[Sequence[int]]:  (Default value = None)
    :param pe_type: PEType:  (Default value = PEType.LePE_ADD)
    :param shape: Sequence[int]:  (Default value = (1)
    :param 192:
    :param 256):
    :param base_arch: str:  (Default value = "tiny-moat-2")
    :param attention_mode: str:  (Default value = "global")
    :param split_head: bool:  (Default value = True)
    :param output_stride: int:  (Default value = 32)
    :param channel_buffer_align: bool:  (Default value = True)
    :param num_blocks: Sequence[int]:  (Default value = (2)
    :param 2):
    :param mbconv_block_expand_ratio: int:  (Default value = 4)
    :param moat_block_expand_ratio: int:  (Default value = 4)
    :param local_window_size: Optional[Sequence[int]]:  (Default value = None)
    :param pe_type: PEType:  (Default value = PEType.LePE_ADD)
    :returns: tiny moat model according to the config.

    """
    assert shape[-2] % 32 == 0
    assert shape[-1] % 32 == 0

    if attention_mode == "global" and local_window_size is not None:
        raise RuntimeError(
            "global attention should not have local_window_size for local attention."
        )

    if output_stride == 32:
        out_stride_stage3, out_stride_stage4 = 16, 32
    else:
        out_stride_stage3, out_stride_stage4 = output_stride, output_stride

    stage_stride = get_stage_strides(output_stride)

    feature_hw = [shape[-2] // output_stride, shape[-1] // output_stride]

    def _get_default_local_window_size(feature_hw):
        """

        :param feature_hw:

        """
        window_hw = []
        attention_field_candidates = [6, 8, 10]
        for h_or_w in feature_hw:
            if h_or_w % attention_field_candidates[0] == 0:
                window_hw.append(attention_field_candidates[0])
            elif h_or_w % attention_field_candidates[1] == 0:
                window_hw.append(attention_field_candidates[1])
            elif h_or_w % attention_field_candidates[2] == 0:
                window_hw.append(attention_field_candidates[2])
            else:
                raise RuntimeError(
                    f"Not a regular feature map size: {feature_hw}, consider other input resolution."
                )
        return window_hw

    if attention_mode == "global":
        window_size = (
            None,
            None,
            [shape[-2] // out_stride_stage3, shape[-1] // out_stride_stage3],
            [shape[-2] // out_stride_stage4, shape[-1] // out_stride_stage4],
        )
    elif attention_mode == "local":
        if local_window_size is None:
            local_window_size = _get_default_local_window_size(feature_hw)
        window_size = (None, None, local_window_size, local_window_size)
    else:
        raise ValueError("Undefined attention mode.")

    if base_arch == "tiny-moat-0":
        tiny_moat_config = MOATConfig(
            num_blocks=num_blocks,
            window_size=window_size,
            attention_mode=attention_mode,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )
    elif base_arch == "tiny-moat-1":
        tiny_moat_config = MOATConfig(
            stem_size=(40, 40),
            hidden_size=(40, 80, 160, 320),
            window_size=window_size,
            attention_mode=attention_mode,
            num_blocks=num_blocks,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )
    elif base_arch == "tiny-moat-2":
        tiny_moat_config = MOATConfig(
            stem_size=(56, 56),
            hidden_size=(56, 112, 224, 448),
            window_size=window_size,
            num_blocks=num_blocks,
            attention_mode=attention_mode,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )

    if channel_buffer_align:
        aligned_hidden_size = [
            math.ceil(h / 32) * 32 for h in tiny_moat_config.hidden_size
        ]
        aligned_stem_size = [math.ceil(h / 32) * 32 for h in tiny_moat_config.stem_size]
        tiny_moat_config.hidden_size = aligned_hidden_size
        tiny_moat_config.stem_size = aligned_stem_size

    logger.info("Using config: %s", tiny_moat_config)
    tiny_moat = MOAT(tiny_moat_config)

    return tiny_moat_config, tiny_moat


if __name__ == "__main__":
    config, model = _build_model()
