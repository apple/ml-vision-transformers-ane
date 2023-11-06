#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""PyTorch MBConv block for MOAT."""
import torch
from torch import nn
from torch.nn import Conv2d
from typing import Optional


class Swish(nn.Module):
    """ """

    def forward(self, x):
        """

        :param x:

        """
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)

    :param block_args: BlockArgs, see above
    :type block_args: namedtuple
    :param global_params: GlobalParam, see above
    :type global_params: namedtuple
    :param name: Block name
    :type name: string

    """

    def __init__(
        self,
        block_args,
        batch_norm_momentum: Optional[float] = 0.99,
        batch_norm_epsilon: Optional[float] = 1e-3,
        drop_rate: Optional[float] = None,
        pre_norm: bool = False,
        name: str = "_block_",
        activation: str = "swish",
    ):
        super(MBConvBlock, self).__init__()
        self.name = name
        self._block_args = block_args
        self.block_activation = activation
        # in torch.batchnorm, (1-momentum)*running_mean + momentum * x_new
        self._bn_mom = 1 - batch_norm_momentum
        self._bn_eps = batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.drop_rate = drop_rate
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.pre_norm = pre_norm

        if self.pre_norm:
            self.pre_norm_layer = nn.BatchNorm2d(
                num_features=self._block_args.input_filters
            )

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = (
            self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            padding="same" if s == 1 else 1,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup,
            out_channels=final_oup,
            kernel_size=1,
            bias=False,
            padding="same",
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        if self.block_activation == "swish":
            self._swish = Swish()
        elif self.block_activation == "relu":
            self._swish = nn.ReLU(inplace=True)
        elif self.block_activation == "gelu":
            self._swish = nn.GELU()
        else:
            raise ValueError("Unsupported activation in MBConv block.")

        if self.drop_rate is not None:
            self.dropout = nn.Dropout(self.drop_rate)

        if block_args.stride == 2:
            self.shortcut_pool = nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            )
        self.shortcut_conv = None
        if block_args.input_filters != block_args.output_filters:
            self.shortcut_conv = Conv2d(
                in_channels=block_args.input_filters,
                out_channels=block_args.output_filters,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=True,
            )

    def forward(self, inputs):
        """param inputs: input tensor (batch_size, C, H, W)
        param drop_connect_rate: drop connect rate (float, between 0 and 1)

        :param inputs:

        """

        shortcut = inputs
        x = inputs

        if self.pre_norm:
            x = self.pre_norm_layer(x)
        # Expansion and Depthwise Convolution
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(x)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if self.id_skip:
            if self._block_args.stride == 1 and input_filters == output_filters:
                if self.drop_rate:
                    x = self.dropout(x)
            elif self._block_args.stride == 2:
                shortcut = self.shortcut_pool(inputs)
                if self.shortcut_conv is not None:
                    shortcut = self.shortcut_conv(shortcut)
            elif (
                self._block_args.stride == 1
                or self._block_args.stride == [1, 1]
                and input_filters != output_filters
            ):
                if self.shortcut_conv is not None:
                    shortcut = self.shortcut_conv(shortcut)
            x = torch.add(x, shortcut)  # skip connection
        return x
