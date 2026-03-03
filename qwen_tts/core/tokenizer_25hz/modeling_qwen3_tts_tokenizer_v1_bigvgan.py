# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3TTSTokenizerV1 model."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers.utils import auto_docstring, logging


from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1DecoderBigVGANConfig,
)
from .modeling_qwen3_tts_tokenizer_v1_bases import (
    Qwen3TTSTokenizerV1DecoderPreTrainedModel,
)

logger = logging.get_logger(__name__)


# Extracted from modeling_qwen3_tts_tokenizer_v1_core.py for better navigation.


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (
            1.0 / (beta + self.no_div_by_zero)
        ) * torch.pow(torch.sin(hidden_states * alpha), 2)

        return hidden_states


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generates a 1D Kaiser-windowed sinc filter.

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(
        kernel_size, beta=beta, periodic=False, dtype=torch.float32
    )

    # Compute time indices
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return torch.zeros(
            (1, 1, kernel_size), dtype=torch.float32
        )  # Ensures correct shape

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]

        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(
            hidden_states, (self.pad_left, self.pad_right), mode="replicate"
        )
        out = F.conv1d(
            hidden_states,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(
            F.pad(x, [self.causal_padding, 0]), self.weight, self.bias
        )


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        causal_type="1",
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                ),
            ]
        )

        if causal_type == "1":
            self.convs2 = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                ]
            )
        else:
            self.convs2 = nn.ModuleList(
                [
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                ]
            )

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        self.activations = nn.ModuleList(
            [
                TorchActivation1d(activation=SnakeBeta(channels))
                for _ in range(self.num_layers)
            ]
        )

        if causal_type == "2":
            self.pre_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, 1),
            )
            self.pre_act = TorchActivation1d(activation=SnakeBeta(channels))
        else:
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x):
        hidden_states = self.pre_conv(x)
        hidden_states = self.pre_act(hidden_states)
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            x = x + hidden_states
        return x


@auto_docstring
class Qwen3TTSTokenizerV1DecoderBigVGANModel(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderBigVGANConfig

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__(config)
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(
            config.mel_dim, config.upsample_initial_channel, 5, 1, padding=2
        )

        # Removing extra ModuleList breaks official state dict
        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes)
            )
        ]
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList(
            [
                AMPBlock(
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    dilation,
                    "1" if layer_idx > 1 else "2",
                )
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                )
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(
                config.upsample_initial_channel // (2**self.num_upsample_layers)
            )
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers),
            1,
            7,
            1,
            padding=3,
            bias=False,
        )

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        return torch.clamp(
            (2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value,
            -max_value,
            max_value,
        )

    def amplitude_to_db(self, amplitude, min_db_level):
        min_level = torch.exp(
            torch.tensor(
                min_db_level / 20.0 * np.log(10),
                device=amplitude.device,
                dtype=amplitude.dtype,
            )
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        amplitude_spectrum = torch.exp(mel_spectrogram)
        decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(decibel_spectrum, 1, -115)

    def forward(self, mel_spectrogram):
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](
                    hidden_representation
                )
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze(1)
