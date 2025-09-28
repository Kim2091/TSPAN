# -*- coding: utf-8 -*-
# Based on SPAN (https://github.com/hongyuanyu/SPAN/blob/main/basicsr/archs/span_arch.py) and TSCUNet (https://github.com/aaf6aa/SCUNet/blob/main/models/network_tscunet.py)
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

## -----------------------------------------------------------------------------
## Helper Functions and Modules from TemporalSPAN
## -----------------------------------------------------------------------------

def _make_pair(value: Any) -> Any:
    """Converts a single integer to a tuple of two identical integers."""
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    """Convolution layer with adaptive padding."""
    kernel_size_t: tuple[int, int] = _make_pair(kernel_size)
    padding = (int((kernel_size_t[0] - 1) / 2), int((kernel_size_t[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(
    act_type: str, inplace: bool = True, neg_slope: float = 0.05, n_prelu: int = 1
) -> nn.Module:
    """Activation functions for ['relu', 'lrelu', 'prelu']."""
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace)
    if act_type == "lrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f"Activation layer [{act_type:s}] is not found.")


def sequential(*args: nn.Module) -> nn.Module:
    """A sequential container that handles nested Sequential modules."""
    if len(args) == 1 and not isinstance(args[0], OrderedDict):
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(submodule for submodule in module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3
) -> nn.Module:
    """Upsampling block using PixelShuffle."""
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    """
    A convolution block that can be re-parameterized into a single 3x3 convolution
    during inference for improved efficiency.
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        s: int = 1,
        bias: Literal[True] = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.stride = s
        self.has_relu = relu
        self.update_params_flag = False
        self.weight_concat = None
        self.bias_concat = None

        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, padding=0, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain1, 1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain1, c_out * gain1, 3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain1, c_out, 1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self) -> None:
        """Merges the weights of the block into a single convolutional layer."""
        w1, b1 = self.conv[0].weight.data, self.conv[0].bias.data
        w2, b2 = self.conv[1].weight.data, self.conv[1].bias.data
        w3, b3 = self.conv[2].weight.data, self.conv[2].bias.data

        # Fuse 1x1, 3x3, 1x1 convs
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        # Fuse skip connection
        sk_w, sk_b = self.sk.weight.data, self.sk.bias.data
        sk_w = F.pad(sk_w, [1, 1, 1, 1])
        self.weight_concat += sk_w
        self.bias_concat += sk_b

        # Set the weights and bias for the evaluation convolution
        self.eval_conv.weight.data.copy_(self.weight_concat)
        self.eval_conv.bias.data.copy_(self.bias_concat)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            if not self.update_params_flag:
                self.update_params()
                self.update_params_flag = True
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """Spatial Attention Block (SPAB)."""
    def __init__(self, in_channels: int, mid_channels: int | None = None, out_channels: int | None = None) -> None:
        super().__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out1 = self.act1(self.c1_r(x))
        out2 = self.act1(self.c2_r(out1))
        out3 = self.c3_r(out2)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att

## -----------------------------------------------------------------------------
## Main TSPAN Class (using TemporalSPAN architecture)
## -----------------------------------------------------------------------------

class TSPAN(nn.Module):
    """
    This class uses the TemporalSPAN architecture.
    """
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        clip_size: int = 5,
        dim: int = 48,
        scale: int = 4,
        state: dict | None = None,
    ) -> None:
        super().__init__()

        # --- Parameter Mapping ---
        num_in_ch = in_nc
        num_out_ch = out_nc
        self.num_frames = clip_size
        self.clip_size = clip_size # Store for external access
        feature_channels = dim
        
        # --- Infer Scale from State Dict (if provided) ---
        upscale = scale
        model_state_to_load = None
        if state:
            model_state_to_load = state.get('params_ema', state)
            if 'upsampler.0.weight' in model_state_to_load:
                upsampler_weight = model_state_to_load['upsampler.0.weight']
                # Formula: out_channels * scale^2 = weight.shape[0]
                inferred_scale = int((upsampler_weight.shape[0] / num_out_ch) ** 0.5)
                
                if inferred_scale != scale:
                    #logger.info(f"INFO: Overriding 'scale' from {scale} to {inferred_scale} based on checkpoint.")
                    upscale = inferred_scale

        # --- Store the final scale factor as a class attribute ---
        self.scale = upscale

        # --- Hardcoded Internal Parameters ---
        bias = True
        norm = False
        img_range = 255.0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        
        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # --- Add 'no_norm' buffer for full checkpoint compatibility ---
        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        # --- Model Architecture (built with the correct 'upscale' value) ---
        self.conv_1 = Conv3XC(self.in_channels * self.num_frames, feature_channels, gain1=2)
        
        self.block_1 = SPAB(feature_channels)
        self.block_2 = SPAB(feature_channels)
        self.block_3 = SPAB(feature_channels)
        self.block_4 = SPAB(feature_channels)
        self.block_5 = SPAB(feature_channels)
        self.block_6 = SPAB(feature_channels)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=bias)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2)

        self.upsampler = pixelshuffle_block(feature_channels, self.out_channels, upscale_factor=upscale)
        
        # --- Load State Dict ---
        if model_state_to_load:
            self.load_state_dict(model_state_to_load, strict=True)

    @property
    def is_norm(self) -> bool:
        """Checks if normalization is enabled."""
        return self.no_norm is None

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes a sequence of frames using the early fusion method.

        Args:
            x: Input tensor with shape (B, T, C, H, W).

        Returns:
            Output tensor with shape (B, C, H_out, W_out).
        """
        b, t, c, h, w = x.shape

        if t != self.num_frames:
            raise ValueError(f"Expected input with {self.num_frames} frames, but received {t}.")

        if self.is_norm:
            self.mean = self.mean.to(x)
            mean_reshaped = self.mean.view(1, 1, c, 1, 1)
            x = (x - mean_reshaped) * self.img_range

        # Early Fusion: Reshape (B, T, C, H, W) -> (B, T*C, H, W)
        x_fused = x.view(b, t * c, h, w)

        # Main forward pass through the network
        out_feature = self.conv_1(x_fused)
        out_b1, _, _ = self.block_1(out_feature)
        out_b2, _, _ = self.block_2(out_b1)
        out_b3, _, _ = self.block_3(out_b2)
        out_b4, _, _ = self.block_4(out_b3)
        out_b5, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _ = self.block_6(out_b5)
        out_b6 = self.conv_2(out_b6)
        
        # Feature concatenation and final upsampling
        out_cat = torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1)
        out = self.conv_cat(out_cat)
        output = self.upsampler(out)


        return output
