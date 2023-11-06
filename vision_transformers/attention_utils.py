#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
from enum import Enum, unique
import logging
from typing import Optional, Sequence

import numpy as np

import torch
from torch import nn
from timm.models.layers import trunc_normal_

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Reference: 
    [1] Swin Transformer: https://arxiv.org/abs/2103.14030
    [2] Swin Github: https://github.com/microsoft/Swin-Transformer
    [3] Local enhanced position embedding: https://arxiv.org/abs/2107.00652 
"""


def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    """Partition image feature map into small windows, in an ANE friendly manner (w/o resorting to 6D tensors).

    :param x: feature map to be partitioned, (batch_size, H, W, C)
    :param window_size: target window_size, (win_h, win_w)
    :param x: torch.Tensor:
    :param window_size: Sequence[int]:
    :returns: (batch_size * num_windows, H, W, C).
    :rtype: Partitioned feature map windows

    """
    B, H, W, C = x.shape
    # example partition process: 1, 12, 16, 160 -> 1, 2, 6, 16, 160 -> 2, 6, 16, 160 -> 2, 6, 2, 8, 160 -> ...
    x = x.reshape(
        (B, H // window_size[0], window_size[0], W, C)
    )  # B, H//w_size, w_size, W, C
    x = x.reshape(
        (B * H // window_size[0], window_size[0], W, C)
    )  # B * H // w_size, w_size, W, C
    x = x.reshape(
        (
            B * H // window_size[0],
            window_size[0],
            W // window_size[1],
            window_size[1],
            -1,
        )
    )
    x = x.permute((0, 2, 1, 3, 4))
    windows = x.reshape((-1, window_size[0], window_size[1], C))
    return windows


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], H: int, W: int):
    """Merge partitioned windows back to feature map

    :param windows: (num_windows*batch_size, win_h, win_w, C)
    :param window_size: Window size
    :type window_size: int
    :param H: Height of image
    :type H: int
    :param W: Width of image
    :type W: int
    :param windows: torch.Tensor:
    :param window_size: Sequence[int]:
    :param H: int:
    :param W: int:
    :returns: (batch_size, H, W, C)
    :rtype: Feature maos

    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.reshape(
        (
            B * H // window_size[0],
            W // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
    )
    x = x.permute((0, 2, 1, 3, 4)).reshape(
        (B * H // window_size[0], window_size[0], W, -1)
    )
    x = x.reshape((B, H // window_size[0], window_size[0], W, -1))
    x = x.reshape((B, H, W, -1))
    return x


@unique
class PEType(Enum):
    """ """

    LePE_ADD = 0
    LePE_FUSED = 1
    RPE = 2
    SINGLE_HEAD_RPE = 3


class WindowAttention(nn.Module):
    """Window/Global based multi-head self attention (MHSA) module

    Supports only non-shifting window attention as there is no native shifting support for ANE.
    Supports attention computation that is efficient on ANE by splitting on the softmax dimension.

    :param dim: Number of input channels.
    :param window_size: The height and width of the window.
    :param num_heads: Number of attention heads.
    :param qkv_bias: If True, add a learnable bias to query.
    :param qk_scale: Override default qk scale of head_dim ** -0.5 if set.
    :param attn_drop: Dropout ratio of attention weight.
    :param proj_drop: Dropout ratio of output.
    :param split_head: Whether to split head for softmax.
            split_softmax reduces latency significantly, therefore enabled by default.
    :param pe_type: position embedding type.

    """

    def __init__(
        self,
        dim: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        split_head: bool = True,
        pe_type: Enum = PEType.LePE_ADD,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.split_head = split_head
        self.pe_type = pe_type

        if pe_type == PEType.RPE or pe_type == PEType.SINGLE_HEAD_RPE:
            # TODO: single-head RPE.
            self.rpe_num_heads = 1 if PEType.SINGLE_HEAD_RPE else num_heads
            logger.info(f"******Using RPE on {self.rpe_num_heads} heads.")
            shape = (
                (2 * window_size[0] - 1),
                (2 * window_size[1] - 1),
                self.rpe_num_heads,
            )

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(shape)
            )  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            # get pair-wise relative position index for each token inside the window
            coords_h = np.arange(self.window_size[0])
            coords_w = np.arange(self.window_size[1])

            mesh = np.meshgrid(coords_h, coords_w)
            # mesh grid returns transposed results compared w/ pytorch
            coords = np.stack((mesh[0].T, mesh[1].T))  # NOTE: 2, Wh, Ww
            coords_flatten = coords.reshape(2, -1)
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose((1, 2, 0))  # Wh*Ww, Wh*Ww, 2

            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            self.relative_position_index = np.sum(relative_coords, -1)  # Wh*Ww, Wh*Ww
            bias_index_bound = shape[0] if len(shape) == 2 else shape[0] * shape[1]
            assert (self.relative_position_index >= 0).all()
            assert (self.relative_position_index < bias_index_bound).all()
        elif pe_type == PEType.LePE_ADD:
            logger.info("******Using LePE_ADD.")
            self.LePE_for_Value = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                groups=dim,
                bias=qkv_bias,
                kernel_size=3,
                padding="same",
            )
            self.abs_pe = nn.Parameter(
                torch.zeros(1, window_size[0] * window_size[1], dim)
            )

        # Use separate conv1x1 projection to avoid L2 cache hit
        self.q_proj = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            bias=qkv_bias,
        )
        self.k_proj = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            bias=qkv_bias,
        )
        self.v_proj = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            bias=qkv_bias,
        )
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # TODO: double check
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """

        :param x: torch.Tensor:

        """
        if self.pe_type == PEType.RPE or self.pe_type == PEType.SINGLE_HEAD_RPE:
            local_table = self.relative_position_bias_table.reshape(
                (-1, self.rpe_num_heads)
            )
        elif self.pe_type == PEType.LePE_ADD:
            x += self.abs_pe

        BW, N, C = x.shape  # BW=num_windows*B=64*1
        assert (
            N == self.window_size[0] * self.window_size[1]
        ), "N: {}, num_windows: {}".format(N, self.window_size[0] * self.window_size[1])
        image_shape = (BW, C, self.window_size[0], self.window_size[1])
        x_2d = x.permute((0, 2, 1)).reshape(image_shape)  # BCHW
        x_flat = torch.unsqueeze(x.permute((0, 2, 1)), 2)  # BC1L

        q, k, v_2d = self.q_proj(x_flat), self.k_proj(x_flat), self.v_proj(x_2d)
        if self.pe_type == PEType.LePE_ADD:
            LePE = self.LePE_for_Value(v_2d).reshape(x_flat.shape)
            mh_LePE = torch.split(LePE, self.dim // self.num_heads, dim=1)
        mh_q = torch.split(q, self.dim // self.num_heads, dim=1)  # BC1L
        mh_v = torch.split(
            v_2d.reshape(x_flat.shape), self.dim // self.num_heads, dim=1
        )
        # BL1C, transposeThenSplit is more efficient than the other way around
        mh_k = torch.split(
            torch.permute(k, (0, 3, 2, 1)), self.dim // self.num_heads, dim=3
        )

        # attn weights in each head.
        attn_weights = [
            torch.einsum("bchq, bkhc->bkhq", qi, ki) * self.scale
            for qi, ki in zip(mh_q, mh_k)
        ]

        # add RPE bias
        if self.pe_type == PEType.RPE or self.pe_type == PEType.SINGLE_HEAD_RPE:
            relative_position_bias = local_table[
                self.relative_position_index.reshape((-1,))
            ].reshape(
                (
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1,
                )
            )  # Wh*Ww, Wh*Ww, nH
            relative_position_bias = torch.unsqueeze(
                relative_position_bias.permute((2, 0, 1)), 2
            )  # nH, Wh*Ww, 1, Wh*Ww
            relative_position_bias = torch.split(relative_position_bias, 1, dim=0)

            # split_softmax
            for head_idx in range(self.num_heads):
                rpe_idx = head_idx if self.pe_type == PEType.RPE else 0
                attn_weights[head_idx] = (
                    attn_weights[head_idx] + relative_position_bias[rpe_idx]
                )

        attn_weights = [
            self.softmax(aw) for aw in attn_weights
        ]  # softmax applied on channel "C"
        mh_w = [self.attn_drop(aw) for aw in attn_weights]

        # compute attn@v
        mh_x = [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(mh_w, mh_v)]
        if self.pe_type == PEType.LePE_ADD:
            mh_x = [v + pe for v, pe in zip(mh_x, mh_LePE)]
        # concat heads
        x = torch.cat(mh_x, dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = torch.squeeze(x, dim=2)
        x = x.permute((0, 2, 1))  # BLC
        return x
