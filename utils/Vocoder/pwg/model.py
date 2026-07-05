"""Minimal Parallel WaveGAN generator used by the bundled checkpoint."""

import logging
import math
from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        self.weight.data.fill_(1.0 / reduce(mul, self.kernel_size, 1))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Stretch2d(torch.nn.Module):
    def __init__(self, time_scale):
        super().__init__()
        self.time_scale = time_scale

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(1, self.time_scale), mode="nearest"
        )


class UpsampleNetwork(torch.nn.Module):
    def __init__(self, upsample_scales):
        super().__init__()
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            self.up_layers.append(Stretch2d(scale))
            self.up_layers.append(
                Conv2d(
                    1,
                    1,
                    kernel_size=(1, scale * 2 + 1),
                    padding=(0, scale),
                    bias=False,
                )
            )

    def forward(self, conditioning):
        conditioning = conditioning.unsqueeze(1)
        for layer in self.up_layers:
            conditioning = layer(conditioning)
        return conditioning.squeeze(1)


class ConvInUpsampleNetwork(torch.nn.Module):
    def __init__(self, upsample_scales, aux_channels, aux_context_window):
        super().__init__()
        self.conv_in = Conv1d(
            aux_channels,
            aux_channels,
            kernel_size=2 * aux_context_window + 1,
            bias=False,
        )
        self.upsample = UpsampleNetwork(upsample_scales)

    def forward(self, conditioning):
        return self.upsample(self.conv_in(conditioning))


class WaveNetResidualBlock(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        residual_channels,
        gate_channels,
        skip_channels,
        aux_channels,
        dropout,
        dilation,
        bias,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            bias=bias,
        )
        self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        gated_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(
            gated_channels, residual_channels, bias=bias
        )
        self.conv1x1_skip = Conv1d1x1(
            gated_channels, skip_channels, bias=bias
        )

    def forward(self, x, conditioning):
        residual = x
        x = self.conv(F.dropout(x, p=self.dropout, training=self.training))
        xa, xb = x.chunk(2, dim=1)
        ca, cb = self.conv1x1_aux(conditioning).chunk(2, dim=1)
        x = torch.tanh(xa + ca) * torch.sigmoid(xb + cb)
        skip = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, skip


class ParallelWaveGANGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        aux_context_window=2,
        dropout=0.0,
        bias=True,
        use_weight_norm=True,
        upsample_net="ConvInUpsampleNetwork",
        upsample_params=None,
        **unused,
    ):
        super().__init__()
        if layers % stacks:
            raise ValueError("layers must be divisible by stacks")
        if upsample_net != "ConvInUpsampleNetwork":
            raise ValueError(f"Unsupported upsample network: {upsample_net}")

        upsample_scales = (upsample_params or {}).get(
            "upsample_scales", [4, 4, 4, 4]
        )
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.upsample_factor = reduce(mul, upsample_scales, 1)
        self.first_conv = Conv1d1x1(
            in_channels, residual_channels, bias=True
        )
        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales, aux_channels, aux_context_window
        )

        layers_per_stack = layers // stacks
        self.conv_layers = torch.nn.ModuleList(
            [
                WaveNetResidualBlock(
                    kernel_size=kernel_size,
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_channels=skip_channels,
                    aux_channels=aux_channels,
                    dropout=dropout,
                    dilation=2 ** (layer % layers_per_stack),
                    bias=bias,
                )
                for layer in range(layers)
            ]
        )
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, noise, conditioning):
        conditioning = self.upsample_net(conditioning)
        if conditioning.size(-1) != noise.size(-1):
            raise ValueError(
                "Upsampled conditioning and noise lengths do not match: "
                f"{conditioning.size(-1)} != {noise.size(-1)}"
            )

        x = self.first_conv(noise)
        skips = 0
        for layer in self.conv_layers:
            x, skip = layer(x, conditioning)
            skips = skips + skip
        x = skips * math.sqrt(1.0 / len(self.conv_layers))
        for layer in self.last_conv_layers:
            x = layer(x)
        return x

    def apply_weight_norm(self):
        def apply(module):
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.weight_norm(module)

        self.apply(apply)

    def remove_weight_norm(self):
        def remove(module):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

        self.apply(remove)
        logging.info("Removed weight normalization from PWG generator.")

