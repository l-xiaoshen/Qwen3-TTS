"""Stateful incremental execution for the Qwen3-TTS 12 Hz decoder.

The convolution and transposed-convolution state decomposition follows the
Apache-2.0 Nari Qwen3-TTS incremental codec implementation, adapted here to
retain the native Transformers KV cache and configured attention backend.
"""

from dataclasses import dataclass, field
from typing import cast

import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .modeling_qwen3_tts_tokenizer_v2_decoder import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
)
from .modeling_qwen3_tts_tokenizer_v2_transformer import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    _call_tensor_module,
)


@dataclass(slots=True)
class Qwen3TTSTokenizerV2DecodeState:
    """Request-local causal state for one 12 Hz codec stream."""

    frame_position: int = 0
    transformer_cache: Cache | None = None
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)
    _decoder_identity: int | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Reset this state to the beginning of a new acoustic stream."""
        self.frame_position = 0
        self.transformer_cache = None
        self.conv_histories.clear()
        self.transconv_overlaps.clear()
        self._decoder_identity = None


class Qwen3TTSTokenizerV2IncrementalDecoder:
    """Decode only fresh codec frames while preserving causal decoder state."""

    def __init__(self, decoder: Qwen3TTSTokenizerV2Decoder) -> None:
        self.decoder = decoder
        self.samples_per_frame = int(decoder.total_upsample)
        self._conv_keys, self._transconv_keys = self._state_keys()

    def _state_keys(self) -> tuple[frozenset[str], frozenset[str]]:
        conv_keys = {"pre_conv", "decoder.0"}
        transconv_keys: set[str] = set()

        for index, _ in enumerate(self.decoder.upsample):
            conv_keys.add(f"upsample.{index}.convnext.dwconv")
            transconv_keys.add(f"upsample.{index}.transconv")

        final_index = len(self.decoder.decoder) - 1
        conv_keys.add(f"decoder.{final_index}")
        for block_index, block in enumerate(
            self.decoder.decoder[1:final_index], start=1
        ):
            if not isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderBlock):
                continue
            transconv_keys.add(f"decoder.{block_index}.transconv")
            for unit_index, _ in enumerate(block.block[2:]):
                conv_keys.add(f"decoder.{block_index}.residual.{unit_index}.conv1")
                conv_keys.add(f"decoder.{block_index}.residual.{unit_index}.conv2")

        return frozenset(conv_keys), frozenset(transconv_keys)

    def _validate_state(self, state: Qwen3TTSTokenizerV2DecodeState) -> None:
        if isinstance(state.frame_position, bool) or state.frame_position < 0:
            raise ValueError("Codec state frame position must be non-negative.")
        if state._decoder_identity not in (None, id(self.decoder)):
            raise ValueError("Codec state belongs to a different decoder instance.")

        if state.frame_position == 0:
            if (
                state.transformer_cache is not None
                or state.conv_histories
                or state.transconv_overlaps
            ):
                raise ValueError("Cold codec state contains unexpected cached tensors.")
            return

        if state.transformer_cache is None:
            raise ValueError("Warm codec state is missing its Transformer cache.")
        if state.transformer_cache.get_seq_length() != state.frame_position:
            raise ValueError("Codec state frame position and KV cache disagree.")
        if state.conv_histories.keys() != self._conv_keys:
            raise ValueError("Warm codec state has incomplete convolution history.")
        if state.transconv_overlaps.keys() != self._transconv_keys:
            raise ValueError(
                "Warm codec state has incomplete transposed-convolution state."
            )

    @staticmethod
    def _causal_conv(
        module: Qwen3TTSTokenizerV2CausalConvNet,
        hidden: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
        key: str,
    ) -> torch.Tensor:
        conv = module.conv
        stride = int(conv.stride[0])
        if stride != 1:
            raise ValueError(
                f"Incremental causal Conv1d requires stride 1, got {stride}."
            )

        history_size = int(module.padding)
        expected_shape = (hidden.shape[1], history_size)
        history = state.conv_histories.get(key)
        if history is None:
            history = hidden.new_zeros(expected_shape)
        elif history.shape != expected_shape:
            raise ValueError(
                f"Invalid codec history {key!r}: expected {expected_shape}, "
                f"got {tuple(history.shape)}."
            )
        elif history.device != hidden.device or history.dtype != hidden.dtype:
            raise ValueError(
                f"Codec history {key!r} does not match the current device and dtype."
            )

        combined = torch.cat((history.unsqueeze(0), hidden), dim=-1)
        output = cast(torch.Tensor, conv(combined))
        if output.shape[-1] != hidden.shape[-1]:
            raise RuntimeError(
                f"Incremental causal Conv1d {key!r} produced "
                f"{output.shape[-1]} positions for {hidden.shape[-1]} inputs."
            )
        retained = (
            combined[0, :, -history_size:] if history_size else combined[0, :, :0]
        )
        state.conv_histories[key] = retained.detach().clone()
        return output.contiguous()

    @staticmethod
    def _causal_transconv(
        module: Qwen3TTSTokenizerV2CausalTransConvNet,
        hidden: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
        key: str,
    ) -> torch.Tensor:
        conv = module.conv
        stride = int(conv.stride[0])
        overlap_size = int(module.right_pad)
        if overlap_size not in (0, stride):
            raise ValueError(
                "Incremental causal ConvTranspose1d requires zero or one stride "
                f"of overlap, got right_pad={overlap_size}, stride={stride}."
            )

        # The carried overlap is bias-free. Bias belongs to each final output
        # position and must be added once, after adjacent frame contributions join.
        expanded = F.conv_transpose1d(
            hidden,
            conv.weight,
            bias=None,
            stride=conv.stride,
            padding=cast(tuple[int, ...], conv.padding),
            output_padding=conv.output_padding,
            groups=conv.groups,
            dilation=conv.dilation,
        )
        if overlap_size:
            expected_shape = (expanded.shape[1], overlap_size)
            overlap = state.transconv_overlaps.get(key)
            if overlap is None:
                overlap = expanded.new_zeros(expected_shape)
            elif overlap.shape != expected_shape:
                raise ValueError(
                    f"Invalid codec overlap {key!r}: expected {expected_shape}, "
                    f"got {tuple(overlap.shape)}."
                )
            elif overlap.device != expanded.device or overlap.dtype != expanded.dtype:
                raise ValueError(
                    f"Codec overlap {key!r} does not match the current device and dtype."
                )
            expanded[:, :, :overlap_size] += overlap.unsqueeze(0)

        emitted_size = hidden.shape[-1] * stride
        emitted = expanded[:, :, :emitted_size]
        if conv.bias is not None:
            emitted = emitted + conv.bias.view(1, -1, 1)
        if overlap_size:
            state.transconv_overlaps[key] = (
                expanded[0, :, emitted_size:].detach().clone()
            )
        else:
            state.transconv_overlaps[key] = expanded.new_empty((expanded.shape[1], 0))
        return emitted.contiguous()

    @classmethod
    def _convnext(
        cls,
        module: Qwen3TTSTokenizerV2ConvNeXtBlock,
        hidden: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
        key: str,
    ) -> torch.Tensor:
        residual = hidden
        hidden = cls._causal_conv(
            module.dwconv, hidden, state, f"{key}.dwconv"
        ).permute(0, 2, 1)
        hidden = _call_tensor_module(module.norm, hidden)
        hidden = _call_tensor_module(module.pwconv1, hidden)
        hidden = _call_tensor_module(module.act, hidden)
        hidden = _call_tensor_module(module.pwconv2, hidden)
        hidden = module.gamma * hidden
        return residual + hidden.permute(0, 2, 1)

    @classmethod
    def _residual_unit(
        cls,
        module: Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
        hidden: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
        key: str,
    ) -> torch.Tensor:
        residual = hidden
        hidden = module.act1(hidden)
        hidden = cls._causal_conv(module.conv1, hidden, state, f"{key}.conv1")
        hidden = module.act2(hidden)
        hidden = cls._causal_conv(module.conv2, hidden, state, f"{key}.conv2")
        return hidden + residual

    def _pre_transformer(
        self,
        hidden: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
    ) -> torch.Tensor:
        output = cast(
            BaseModelOutputWithPast,
            self.decoder.pre_transformer(
                inputs_embeds=hidden,
                past_key_values=state.transformer_cache,
                use_cache=True,
            ),
        )
        output_hidden = output.last_hidden_state
        if not isinstance(output_hidden, torch.Tensor):
            raise TypeError("Codec Transformer did not return tensor hidden states.")
        cache = output.past_key_values
        if not isinstance(cache, Cache):
            raise TypeError("Codec Transformer did not return a native KV cache.")
        state.transformer_cache = cache
        return output_hidden

    def _decode_one(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
    ) -> torch.Tensor:
        self._validate_state(state)
        if codes.shape[0] != 1:
            raise ValueError("Each incremental codec state owns exactly one batch row.")
        if codes.shape[-1] == 0:
            return next(self.decoder.parameters()).new_empty((1, 1, 0))

        original_position = state.frame_position
        state._decoder_identity = id(self.decoder)

        hidden = self.decoder.quantizer.decode(codes)
        hidden = self._causal_conv(
            self.decoder.pre_conv, hidden, state, "pre_conv"
        ).transpose(1, 2)
        hidden = self._pre_transformer(hidden, state).permute(0, 2, 1)

        for upsample_index, blocks in enumerate(self.decoder.upsample):
            typed_blocks = cast(torch.nn.ModuleList, blocks)
            hidden = self._causal_transconv(
                cast(Qwen3TTSTokenizerV2CausalTransConvNet, typed_blocks[0]),
                hidden,
                state,
                f"upsample.{upsample_index}.transconv",
            )
            hidden = self._convnext(
                cast(Qwen3TTSTokenizerV2ConvNeXtBlock, typed_blocks[1]),
                hidden,
                state,
                f"upsample.{upsample_index}.convnext",
            )

        wav = self._causal_conv(
            cast(Qwen3TTSTokenizerV2CausalConvNet, self.decoder.decoder[0]),
            hidden,
            state,
            "decoder.0",
        )
        final_index = len(self.decoder.decoder) - 1
        for block_index, block in enumerate(
            self.decoder.decoder[1:final_index], start=1
        ):
            if not isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderBlock):
                wav = block(wav)
                continue
            wav = _call_tensor_module(block.block[0], wav)
            wav = self._causal_transconv(
                cast(Qwen3TTSTokenizerV2CausalTransConvNet, block.block[1]),
                wav,
                state,
                f"decoder.{block_index}.transconv",
            )
            for unit_index, unit in enumerate(block.block[2:]):
                if not isinstance(unit, Qwen3TTSTokenizerV2DecoderDecoderResidualUnit):
                    raise TypeError(
                        "Codec decoder block contains an unexpected module."
                    )
                wav = self._residual_unit(
                    unit,
                    wav,
                    state,
                    f"decoder.{block_index}.residual.{unit_index}",
                )
        wav = self._causal_conv(
            cast(
                Qwen3TTSTokenizerV2CausalConvNet,
                self.decoder.decoder[final_index],
            ),
            wav,
            state,
            f"decoder.{final_index}",
        )

        fresh_frames = int(codes.shape[-1])
        expected_position = original_position + fresh_frames
        cache = state.transformer_cache
        if cache is None or cache.get_seq_length() != expected_position:
            raise RuntimeError("Codec Transformer cache advanced by an invalid amount.")
        state.frame_position = expected_position

        expected_samples = fresh_frames * self.samples_per_frame
        if wav.shape[-1] != expected_samples:
            raise RuntimeError(
                f"Incremental codec produced {wav.shape[-1]} samples, "
                f"expected {expected_samples}."
            )
        return wav.clamp(min=-1, max=1)

    @torch.inference_mode()
    def __call__(
        self,
        codes: torch.Tensor,
        states: list[Qwen3TTSTokenizerV2DecodeState],
    ) -> torch.Tensor:
        """Decode ``(batch, codebooks, fresh_frames)`` into fresh samples."""
        if codes.ndim != 3:
            raise ValueError(
                f"Expected codec codes with rank 3, got shape {tuple(codes.shape)}."
            )
        if codes.dtype != torch.long:
            raise TypeError("Codec codes must use torch.long dtype.")
        if codes.shape[0] != len(states):
            raise ValueError(
                f"State count {len(states)} does not match batch {codes.shape[0]}."
            )
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} codebooks, "
                f"got {codes.shape[1]}."
            )
        if len(states) == 0:
            raise ValueError("Incremental codec requires at least one stream.")

        return torch.cat(
            [
                self._decode_one(codes[index : index + 1], state)
                for index, state in enumerate(states)
            ],
            dim=0,
        )


__all__ = [
    "Qwen3TTSTokenizerV2DecodeState",
    "Qwen3TTSTokenizerV2IncrementalDecoder",
]
