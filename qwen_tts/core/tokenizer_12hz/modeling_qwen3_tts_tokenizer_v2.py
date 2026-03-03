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
"""PyTorch Qwen3TTSTokenizerV2 model."""

from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, logging

from .configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .modeling_qwen3_tts_tokenizer_v2_core import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2Encoder,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: List[torch.LongTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2DecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: List[torch.FloatTensor] = None


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
    def __init__(self, config: Qwen3TTSTokenizerV2Config):
        super().__init__(config)
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(
            self.config.encoder_config
        )
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(
            self.config.decoder_config
        )

        self.post_init()

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerV2EncoderOutput
    ]:
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

        encoded_frames = self.encoder.encode(
            input_values=input_values.unsqueeze(1), return_dict=True
        )
        audio_codes = encoded_frames.audio_codes[:, : self.encoder_valid_num_quantizers]
        audio_codes = [
            code[..., : -(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1)
            for code, mask in zip(audio_codes, padding_mask)
        ]

        if not return_dict:
            return (audio_codes,)

        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes)

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Qwen3TTSTokenizerV2DecoderOutput]:
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
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(
            1
        )

        audio_values = [
            audio[:length] for audio, length in zip(audio_values, audio_lengths)
        ]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV2DecoderOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerV2Model", "Qwen3TTSTokenizerV2PreTrainedModel"]
