"""PyTorch Qwen3TTSTokenizerV2 model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mimi.configuration_mimi import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.utils import logging
from typing_extensions import override

from .configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from .modeling_qwen3_tts_tokenizer_v2_transformer import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2DecoderPreTrainedModel,
    Qwen3TTSTokenizerV2DecoderTransformerModel,
    _call_tensor_module,
)

if TYPE_CHECKING:
    from .modeling_qwen3_tts_tokenizer_v2_incremental import (
        Qwen3TTSTokenizerV2DecodeState,
        Qwen3TTSTokenizerV2IncrementalDecoder,
    )

logger = logging.get_logger(__name__)


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

    def __init__(self, in_features: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    @override
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


class Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1) -> None:
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3TTSTokenizerV2CausalConvNet(
            dim, dim, kernel_size=7, dilation=dilation
        )
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3TTSTokenizerV2CausalConvNet(dim, dim, kernel_size=1)

    @override
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state

        hidden_state = _call_tensor_module(self.act1, hidden_state)
        hidden_state = _call_tensor_module(self.conv1, hidden_state)
        hidden_state = _call_tensor_module(self.act2, hidden_state)
        hidden_state = _call_tensor_module(self.conv2, hidden_state)
        return hidden_state + residual


class Qwen3TTSTokenizerV2DecoderDecoderBlock(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    def __init__(
        self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx: int
    ) -> None:
        super().__init__(config)
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block: list[nn.Module] = [
            SnakeBeta(in_dim),
            Qwen3TTSTokenizerV2CausalTransConvNet(
                in_dim, out_dim, 2 * upsample_rate, upsample_rate
            ),
        ]

        for dilation in (1, 3, 9):
            block.append(
                Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(out_dim, dilation)
            )

        self.block = nn.ModuleList(block)

    @override
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden = _call_tensor_module(block, hidden)
        return hidden


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon

        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = (
            self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        )
        quantized = F.embedding(codes, embedding)
        return quantized


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim

        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = _call_tensor_module(self.project_out, quantized)
        quantized = quantized.transpose(1, 2)
        return quantized


class ResidualVectorQuantization(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VectorQuantization(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    epsilon=epsilon,
                )
                for _ in range(num_quantizers)
            ]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = cast(VectorQuantization, self.layers[idx])
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: int | None = None,
        output_dimension: int | None = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ) -> None:
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )
        self.vq = ResidualVectorQuantization(
            dim=self.dimension, codebook_size=self.bins, num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = _call_tensor_module(self.output_proj, quantized)
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        dimension: int = 128,
        input_dimension: int | None = None,
        output_dimension: int | None = None,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        if n_q <= n_q_semantic:
            raise ValueError(
                f"Number of quantizers {n_q} must be larger "
                f"than the number of semantic quantizers {n_q_semantic}."
            )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q_semantic,
            q_dropout=False,
            no_quantization_rate=no_quantization_rate,
            bins=bins,
            decay=decay,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - n_q_semantic,
            q_dropout=q_dropout,
            no_quantization_rate=no_quantization_rate,
            bins=bins,
            decay=decay,
            force_projection=True,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class Qwen3TTSTokenizerV2Decoder(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig) -> None:
        super().__init__(config)
        self._incremental_decoder: Qwen3TTSTokenizerV2IncrementalDecoder | None = None
        self.total_upsample = int(
            np.prod((*config.upsample_rates, *config.upsampling_ratios))
        )
        self.pre_transformer = Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(
            config
        )

        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerV2CausalConvNet(
            config.codebook_dim,
            config.latent_dim,
            kernel_size=3,
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerV2CausalTransConvNet(
                            config.latent_dim, config.latent_dim, factor, factor
                        ),
                        Qwen3TTSTokenizerV2ConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder: list[nn.Module] = [
            Qwen3TTSTokenizerV2CausalConvNet(config.latent_dim, config.decoder_dim, 7)
        ]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerV2DecoderDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3TTSTokenizerV2CausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)

        self.post_init()

    def _get_incremental_decoder(
        self,
    ) -> Qwen3TTSTokenizerV2IncrementalDecoder:
        from .modeling_qwen3_tts_tokenizer_v2_incremental import (
            Qwen3TTSTokenizerV2IncrementalDecoder,
        )

        if self._incremental_decoder is None:
            self._incremental_decoder = Qwen3TTSTokenizerV2IncrementalDecoder(self)
        return self._incremental_decoder

    @override
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}"
            )

        hidden = self.quantizer.decode(codes)
        hidden = _call_tensor_module(self.pre_conv, hidden).transpose(1, 2)

        transformer_output = cast(
            BaseModelOutputWithPast, self.pre_transformer(inputs_embeds=hidden)
        )
        hidden = transformer_output.last_hidden_state
        if hidden is None:
            raise RuntimeError("Decoder transformer did not return hidden states.")
        hidden = hidden.permute(0, 2, 1)
        for blocks_module in self.upsample:
            for block in cast(nn.ModuleList, blocks_module):
                hidden = _call_tensor_module(block, hidden)
        wav = hidden
        for block in self.decoder:
            wav = _call_tensor_module(block, wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """Decode bounded chunks with exact request-local causal state.

        ``left_context_size`` remains for call compatibility. Stateful decoding
        derives the complete required context from the model instead of replaying
        an approximate fixed number of input frames.
        """
        if chunk_size <= 0:
            raise ValueError("`chunk_size` must be positive.")
        if left_context_size < 0:
            raise ValueError("`left_context_size` must be non-negative.")
        if codes.ndim != 3:
            raise ValueError("`codes` must have shape (batch, codebooks, frames).")

        from .modeling_qwen3_tts_tokenizer_v2_incremental import (
            Qwen3TTSTokenizerV2DecodeState,
        )

        if codes.shape[-1] == 0:
            return next(self.parameters()).new_empty((codes.shape[0], 1, 0))

        incremental_decoder = self._get_incremental_decoder()
        states = [Qwen3TTSTokenizerV2DecodeState() for _ in range(codes.shape[0])]
        wavs: list[torch.Tensor] = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            wavs.append(incremental_decoder(codes[..., start_index:end_index], states))
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    @staticmethod
    def new_decode_state() -> Qwen3TTSTokenizerV2DecodeState:
        """Create independent state for one incremental codec stream."""
        from .modeling_qwen3_tts_tokenizer_v2_incremental import (
            Qwen3TTSTokenizerV2DecodeState,
        )

        return Qwen3TTSTokenizerV2DecodeState()

    def incremental_decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
    ) -> torch.Tensor:
        """Decode fresh ``(1, codebooks, frames)`` codes and advance state."""
        return self._get_incremental_decoder()(codes, [state])


class Qwen3TTSTokenizerV2Encoder(MimiModel):
    def __init__(self, config: MimiConfig) -> None:
        super().__init__(config)
        self.config = config

        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None

        self.post_init()
