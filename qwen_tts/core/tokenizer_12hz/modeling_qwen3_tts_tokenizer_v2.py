"""PyTorch Qwen3TTSTokenizerV2 model."""

from dataclasses import dataclass, field
from typing import cast

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput
from transformers.utils import ModelOutput, auto_docstring, logging

from .configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .modeling_qwen3_tts_tokenizer_v2_core import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2Encoder,
)
from .modeling_qwen3_tts_tokenizer_v2_incremental import (
    Qwen3TTSTokenizerV2DecodeState,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: list[torch.LongTensor] = field(default_factory=list)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2DecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: list[torch.Tensor] = field(default_factory=list)


@auto_docstring
class Qwen3TTSTokenizerV2PreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV2 model.
    """
)
class Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2PreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV2Config) -> None:
        super().__init__(config)
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = cast(
            Qwen3TTSTokenizerV2Encoder,
            Qwen3TTSTokenizerV2Encoder._from_config(self.config.encoder_config),
        )
        self.decoder = cast(
            Qwen3TTSTokenizerV2Decoder,
            Qwen3TTSTokenizerV2Decoder._from_config(self.config.decoder_config),
        )

        self.post_init()

    def get_model_type(self) -> str:
        return self.config.model_type

    def get_input_sample_rate(self) -> int:
        return self.input_sample_rate

    def get_output_sample_rate(self) -> int:
        return self.output_sample_rate

    def get_encode_downsample_rate(self) -> int:
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self) -> int:
        return self.decode_upsample_rate

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple[list[torch.LongTensor]] | Qwen3TTSTokenizerV2EncoderOutput:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        if padding_mask is None:
            raise ValueError("`padding_mask` is required for encode.")

        encoded_frames = cast(
            MimiEncoderOutput,
            self.encoder.encode(
                input_values=input_values.unsqueeze(1), return_dict=True
            ),
        )
        encoded_audio_codes = encoded_frames.audio_codes
        if encoded_audio_codes is None:
            raise RuntimeError("Mimi encoder did not return audio codes.")
        encoded_audio_codes = encoded_audio_codes[
            :, : self.encoder_valid_num_quantizers
        ]
        audio_codes: list[torch.LongTensor] = [
            cast(
                torch.LongTensor,
                code[
                    ..., : -(-int(mask.sum().item()) // self.encode_downsample_rate)
                ].transpose(0, 1),
            )
            for code, mask in zip(encoded_audio_codes, padding_mask)
        ]

        if not return_dict:
            return (audio_codes,)

        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes=audio_codes)

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[list[torch.Tensor]] | Qwen3TTSTokenizerV2DecoderOutput:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)
        decoded_audio = self.decoder.chunked_decode(
            audio_codes.transpose(1, 2)
        ).squeeze(1)

        audio_values: list[torch.Tensor] = [
            audio[:length] for audio, length in zip(decoded_audio, audio_lengths)
        ]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV2DecoderOutput(audio_values=audio_values)

    @staticmethod
    def new_decode_state() -> Qwen3TTSTokenizerV2DecodeState:
        """Create request-local state for an incremental 12 Hz decode stream."""
        return Qwen3TTSTokenizerV2DecodeState()

    @torch.inference_mode()
    def decode_incremental(
        self,
        audio_codes: torch.Tensor,
        state: Qwen3TTSTokenizerV2DecodeState,
        return_dict: bool | None = None,
    ) -> tuple[list[torch.Tensor]] | Qwen3TTSTokenizerV2DecoderOutput:
        """Decode only fresh, unpadded frames and advance ``state`` in place."""
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        if audio_codes.ndim != 3:
            raise ValueError(
                "`audio_codes` must have shape (1, fresh_frames, num_quantizers)."
            )
        if audio_codes.shape[0] != 1:
            raise ValueError("One incremental codec state accepts one batch row.")
        if audio_codes.shape[2] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} codebooks, "
                f"got {audio_codes.shape[2]}."
            )
        if audio_codes.dtype != torch.long:
            raise TypeError("`audio_codes` must use torch.long dtype.")
        decoded_audio = self.decoder.incremental_decode(
            audio_codes.transpose(1, 2), state
        ).squeeze(1)
        audio_values = [decoded_audio[0]]
        if not return_dict:
            return (audio_values,)
        return Qwen3TTSTokenizerV2DecoderOutput(audio_values=audio_values)


__all__ = [
    "Qwen3TTSTokenizerV2DecodeState",
    "Qwen3TTSTokenizerV2Model",
    "Qwen3TTSTokenizerV2PreTrainedModel",
]
