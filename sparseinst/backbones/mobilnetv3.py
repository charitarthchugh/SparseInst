# MobileNetV3 implementation for Detectron2 backbone
# Based on torchvision's MobileNetV3 implementation

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import BatchNorm2d
from functools import partial
from collections.abc import Sequence
from typing import Any, Callable, Optional, List

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone


def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block implementation"""

    def __init__(
        self, input_channels, squeeze_channels, scale_activation=nn.Hardsigmoid
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = scale_activation(inplace=True)

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x


class ConvBNActivation(nn.Sequential):
    """Conv2d + BatchNorm + Activation layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        norm_layer=BatchNorm2d,
        activation_layer=None,
        bias=False,
    ):
        padding = (kernel_size - 1) // 2 * dilation
        layers = [
            Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)


class InvertedResidualConfig:
    """Configuration class for InvertedResidual"""

    def __init__(
        self,
        input_channels,
        kernel,
        expanded_channels,
        out_channels,
        use_se,
        activation,
        stride,
        dilation,
        width_mult,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # HS: Hardswish, RE: ReLU
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels, width_mult):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """InvertedResidual block implementation"""

    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(
            SqueezeExcitation, scale_activation=nn.Hardsigmoid
        ),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvBNActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvBNActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze-excitation
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvBNActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(Backbone):
    """MobileNetV3 backbone for Detectron2"""

    def __init__(
        self,
        cfg,
        arch="mobilenet_v3_large",
        width_mult=1.0,
        reduced_tail=False,
        dilated=False,
        input_size=224,
    ):
        super().__init__()

        # Define feature extraction points and feature info
        self.return_features_indices = []
        self.return_features_num_channels = []

        # Create inverted_residual_setting
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(
            InvertedResidualConfig.adjust_channels, width_mult=width_mult
        )

        if arch == "mobilenet_v3_large":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(
                    112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation
                ),  # C4
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    dilation,
                ),
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    dilation,
                ),
            ]
            last_channel = adjust_channels(1280 // reduce_divider)  # C5

            # Feature extraction points (layer indices) for large model
            self.return_features_indices = [
                1,
                3,
                6,
                12,
                15,
            ]  # Layers corresponding to C1-C5

        elif arch == "mobilenet_v3_small":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(
                    48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation
                ),  # C4
                bneck_conf(
                    96 // reduce_divider,
                    5,
                    576 // reduce_divider,
                    96 // reduce_divider,
                    True,
                    "HS",
                    1,
                    dilation,
                ),
                bneck_conf(
                    96 // reduce_divider,
                    5,
                    576 // reduce_divider,
                    96 // reduce_divider,
                    True,
                    "HS",
                    1,
                    dilation,
                ),
            ]
            last_channel = adjust_channels(1024 // reduce_divider)  # C5

            # Feature extraction points (layer indices) for small model
            self.return_features_indices = [
                0,
                1,
                3,
                8,
                11,
            ]  # Layers corresponding to C1-C5

        else:
            raise ValueError(f"Unsupported model type {arch}")

        # Building the network
        norm_layer = partial(BatchNorm2d, eps=0.001, momentum=0.01)
        block = InvertedResidual

        self.features = nn.ModuleList()

        # First layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.features.append(
            ConvBNActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # Inverted residual blocks
        for i, cnf in enumerate(inverted_residual_setting):
            self.features.append(block(cnf, norm_layer))
            if (
                i + 1 in self.return_features_indices
            ):  # +1 because we already added the first conv
                self.return_features_num_channels.append(cnf.out_channels)

        # Final conv layer
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.features.append(
            ConvBNActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        if len(self.features) - 1 in self.return_features_indices:
            self.return_features_num_channels.append(lastconv_output_channels)

        # Initialize weights
        self._initialize_weights()

        # Freeze backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

        # Setup feature info for Detectron2
        if arch == "mobilenet_v3_large":
            self._out_feature_channels = {
                "res2": 24,  # C1, stride 4
                "res3": 40,  # C2, stride 8
                "res4": 80,  # C3, stride 16
                "res5": 160,  # C4, stride 32
                "res6": last_channel,  # C5, final feature
            }
            self._out_feature_strides = {
                "res2": 4,
                "res3": 8,
                "res4": 16,
                "res5": 32,
                "res6": 32,
            }
        else:  # small
            self._out_feature_channels = {
                "res2": 16,  # C1, stride 4
                "res3": 24,  # C2, stride 8
                "res4": 40,  # C3, stride 16
                "res5": 96 // reduce_divider,  # C4, stride 32
                "res6": last_channel,  # C5, final feature
            }
            self._out_feature_strides = {
                "res2": 4,
                "res3": 8,
                "res4": 16,
                "res5": 32,
                "res6": 32,
            }

        # Define output features
        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    def _freeze_backbone(self, freeze_at):
        """Freeze layers up to specified layer"""
        for layer_index in range(min(freeze_at, len(self.features))):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        """Forward pass implementation with feature extraction at specified layers"""
        res = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.return_features_indices:
                res.append(x)

        # Return features in format expected by Detectron2
        output = {}
        for i, feature in enumerate(res):
            output[f"res{i + 2}"] = feature

        return {k: v for k, v in output.items() if k in self._out_features}

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_backbone(cfg, input_shape):
    """
    Create a MobileNetV3 instance from config.

    Returns:
        MobileNetV3: a :class:`MobileNetV3` backbone instance.
    """
    # Get model configuration from cfg
    arch = (
        cfg.MODEL.MOBILENETV3.ARCH
        if hasattr(cfg.MODEL, "MOBILENETV3") and hasattr(cfg.MODEL.MOBILENETV3, "ARCH")
        else "mobilenet_v3_large"
    )
    width_mult = (
        cfg.MODEL.MOBILENETV3.WIDTH_MULT
        if hasattr(cfg.MODEL, "MOBILENETV3")
        and hasattr(cfg.MODEL.MOBILENETV3, "WIDTH_MULT")
        else 1.0
    )
    reduced_tail = (
        cfg.MODEL.MOBILENETV3.REDUCED_TAIL
        if hasattr(cfg.MODEL, "MOBILENETV3")
        and hasattr(cfg.MODEL.MOBILENETV3, "REDUCED_TAIL")
        else False
    )
    dilated = (
        cfg.MODEL.MOBILENETV3.DILATED
        if hasattr(cfg.MODEL, "MOBILENETV3")
        and hasattr(cfg.MODEL.MOBILENETV3, "DILATED")
        else False
    )
    input_size = 224  # Default input size

    model = MobileNetV3(
        cfg,
        arch=arch,
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        dilated=dilated,
        input_size=input_size,
    )

    return model
