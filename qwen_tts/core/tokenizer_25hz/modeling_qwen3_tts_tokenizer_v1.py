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

from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, logging
from transformers.utils.hub import cached_file

from torch.nn.utils.rnn import pad_sequence

from .vq.whisper_encoder import get_mel_audio, get_T_after_cnn
from .vq.speech_vq import WhisperEncoderVQ, XVectorExtractor

from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV1EncoderConfig,
    Qwen3TTSTokenizerV1DecoderConfig,
)
from .modeling_qwen3_tts_tokenizer_v1_core import (
    Qwen3TTSTokenizerV1DecoderBigVGANModel,
    Qwen3TTSTokenizerV1DecoderDiTModel,
    Qwen3TTSTokenizerV1DecoderPreTrainedModel,
    Qwen3TTSTokenizerV1EncoderPreTrainedModel,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i,).
    xvectors (`List[torch.FloatTensor]`):
        X-vector embeddings computed using `model.encode`, each tensor has shape (xvector_dim,).
    ref_mels (`List[torch.FloatTensor]`):
        Reference mel spectrogram computed using `model.encode`, each tensor has shape (mel_length_i, mel_dim,).
    """

    audio_codes: List[torch.LongTensor] = None
    xvectors: List[torch.FloatTensor] = None
    ref_mels: List[torch.FloatTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1DecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: List[torch.FloatTensor] = None


@auto_docstring
class Qwen3TTSTokenizerV1Decoder(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    _no_split_modules = [
        "Qwen3TTSTokenizerV1DecoderDiTModel",
        "Qwen3TTSTokenizerV1DecoderBigVGANModel",
    ]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen3TTSTokenizerV1Decoder will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.dit = Qwen3TTSTokenizerV1DecoderDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.bigvgan = Qwen3TTSTokenizerV1DecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generates a waveform from input code and conditioning parameters."""

        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = self.bigvgan(mel_spectrogram)

        return waveform


class Qwen3TTSTokenizerV1Encoder(Qwen3TTSTokenizerV1EncoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1EncoderConfig

    def __init__(self, config: Qwen3TTSTokenizerV1EncoderConfig):
        super().__init__(config)

        self.tokenizer = WhisperEncoderVQ(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_window=config.n_window,
            output_dim=config.output_dim,
            grad_checkpointing=config.grad_checkpointing,
            enable_mp=config.enable_mp,
            audio_sequence_parallel=config.audio_sequence_parallel,
            audio_vq_type=config.audio_vq_type,
            audio_vq_layers=config.audio_vq_layers,
            audio_vq_codebook_size=config.audio_vq_codebook_size,
            audio_vq_codebook_dim=config.audio_vq_codebook_dim,
            audio_vq_pe=config.audio_vq_pe,
            audio_vq_ds_rate=config.audio_vq_ds_rate,
        )

        self.padding = True
        self.audio_vq_ds_rate = self.tokenizer.audio_vq_ds_rate

    def speech2mel(self, speechs):
        mels = [
            get_mel_audio(
                speech, padding=self.padding, audio_vq_ds_rate=self.audio_vq_ds_rate
            )
            .to(speech.dtype)
            .to(self.tokenizer.conv1.weight.device)
            for speech in speechs
        ]
        return mels

    def mel2code(self, mels):
        audio_mellens = [mel.size(-1) for mel in mels]
        audio_aftercnnlens = [get_T_after_cnn(T) for T in audio_mellens]
        audio_seqlens = [T + 2 for T in audio_aftercnnlens]

        with torch.no_grad():
            _, indices = self.tokenizer(
                x_list=mels,
                audio_mellens=audio_mellens,
                audio_aftercnnlens=audio_aftercnnlens,
                audio_seqlens=audio_seqlens,
                return_indices=True,
            )

        indice_lens = [T // self.tokenizer.audio_vq_ds_rate for T in audio_aftercnnlens]
        indices = pad_sequence(
            torch.split(indices, indice_lens), batch_first=True, padding_value=0
        )

        return indices, indice_lens

    def quantize_speech(self, speechs):
        mels = self.speech2mel(speechs)
        indices, indice_lens = self.mel2code(mels)
        return indices, indice_lens


@auto_docstring
class Qwen3TTSTokenizerV1PreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV1 model.
    """
)
class Qwen3TTSTokenizerV1Model(Qwen3TTSTokenizerV1PreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV1Config):
        super().__init__(config)
        self.config = config

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerV1Encoder._from_config(
            self.config.encoder_config
        )
        self.decoder = Qwen3TTSTokenizerV1Decoder._from_config(
            self.config.decoder_config
        )

        self.encoder_xvector_extractor = None

        self.post_init()

    def load_encoder_xvector_extractor(self, model_path):
        self.encoder_xvector_extractor = XVectorExtractor(model_path)

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        encoder_xvector_extractor_path = cached_file(
            pretrained_model_name_or_path,
            "campplus.onnx",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if encoder_xvector_extractor_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{encoder_xvector_extractor_path} not exists"""
            )
        model.load_encoder_xvector_extractor(encoder_xvector_extractor_path)

        return model

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerV1EncoderOutput
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

        wavs = [value[: mask.sum()] for value, mask in zip(input_values, padding_mask)]

        codes, codes_lens = self.encoder.quantize_speech(wavs)
        codes = [code[:length] for code, length in zip(codes, codes_lens)]

        xvectors = []
        ref_mels = []
        for wav in wavs:
            xvector, ref_mel = self.encoder_xvector_extractor.extract_code(
                wav.cpu().numpy()
            )
            xvector = torch.tensor(xvector).to(wav.dtype).to(wav.device)
            ref_mel = torch.tensor(ref_mel).to(wav.dtype).to(wav.device)
            xvectors.append(xvector)
            ref_mels.append(ref_mel)

        if not return_dict:
            return (codes, xvectors, ref_mels)

        return Qwen3TTSTokenizerV1EncoderOutput(codes, xvectors, ref_mels)

    def decode(
        self,
        audio_codes: torch.Tensor,
        xvectors: torch.Tensor,
        ref_mels: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Qwen3TTSTokenizerV1DecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            xvectors (`torch.FloatTensor` of shape `(batch_size, xvector_dim)`, *optional*):
                X-vector embeddings computed using `model.encode`.
            ref_mels (`torch.FloatTensor` of shape `(batch_size, mel_length, mel_dim)`, *optional*):
                Reference mel spectrogram computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        audio_lengths = (audio_codes > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)
        audio_values = self.decoder(
            code=audio_codes, reference_mel=ref_mels, conditioning=xvectors
        )

        audio_values = [
            audio[:length] for audio, length in zip(audio_values, audio_lengths)
        ]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV1DecoderOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerV1Model", "Qwen3TTSTokenizerV1PreTrainedModel"]
