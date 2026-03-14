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
"""Base conditional-generation model for Qwen3 TTS."""

import json
import os

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.utils import logging
from transformers.utils.hub import cached_file

from ...inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from .modeling_qwen3_tts_core import (
    Qwen3TTSPreTrainedModel,
    Qwen3TTSSpeakerEncoder,
    mel_spectrogram,
)
from .modeling_qwen3_tts_generation import Qwen3TTSGenerationMixin
from .modeling_qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from .modeling_qwen3_tts_types import (
    GenerateConfigValue,
    VoiceClonePrompt,
)

logger = logging.get_logger(__name__)


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub. Users can specify the
    allow_patterns to download only the necessary weights.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE

    for allow_pattern in allow_patterns:
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_pattern,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_only,
        )
    return hf_folder


class Qwen3TTSConditionalGenerationBase(
    Qwen3TTSGenerationMixin, Qwen3TTSPreTrainedModel
):
    config_class = Qwen3TTSConfig

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config

        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)

        if config.tts_model_type == "base":
            self.speaker_encoder: Qwen3TTSSpeakerEncoder | None = (
                Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
            )
        else:
            self.speaker_encoder = None

        self.speech_tokenizer: Qwen3TTSTokenizer | None = None
        self.generate_config: dict[str, GenerateConfigValue] | None = None

        supported_speakers = self.config.talker_config.spk_id or {}
        supported_languages = self.config.talker_config.codec_language_id or {}
        self.supported_speakers = list(supported_speakers.keys())
        self.supported_languages = ["Auto"]
        for language_id in supported_languages.keys():
            if "dialect" not in language_id:
                self.supported_languages.append(language_id)

        self.speaker_encoder_sample_rate = (
            self.config.speaker_encoder_config.sample_rate
        )
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type

        self.post_init()

    def load_speech_tokenizer(self, speech_tokenizer: Qwen3TTSTokenizer) -> None:
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(
        self, generate_config: dict[str, GenerateConfigValue]
    ) -> None:
        self.generate_config = generate_config

    def get_supported_speakers(self) -> list[str]:
        return self.supported_speakers

    def get_supported_languages(self) -> list[str]:
        return self.supported_languages

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
        # Hotfix to enable passing the correct attn implementation which is stored in the config but not in kwargs
        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if (
            requested_attn_implementation is None
            and config
            and config._attn_implementation
        ):
            requested_attn_implementation = config._attn_implementation

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
            attn_implementation=requested_attn_implementation,
            **kwargs,
        )
        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            download_cache_dir = kwargs.get("cache_dir", cache_dir)
            download_revision = kwargs.get("revision", revision)
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=download_cache_dir,
                allow_patterns=["speech_tokenizer/*"],
                revision=download_revision,
            )
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if speech_tokenizer_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{speech_tokenizer_path} not exists"""
            )
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            *model_args,
            **kwargs,
        )
        model.load_speech_tokenizer(speech_tokenizer)

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if generate_config_path is None:
            raise ValueError(
                f"{pretrained_model_name_or_path}/generation_config.json not exists"
            )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            generate_config = json.load(f)
        model.load_generate_config(generate_config)

        return model

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        assert sr == 24000, "Only support 24kHz audio"
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mels = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        speaker_encoder = self.speaker_encoder
        if speaker_encoder is None:
            raise RuntimeError("Speaker encoder is not available for this model.")
        speaker_embedding = speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return speaker_embedding

    @torch.inference_mode()
    def generate_speaker_prompt(self, ref_spk_embedding: torch.Tensor) -> torch.Tensor:
        return ref_spk_embedding.to(self.talker.device).to(self.talker.dtype)

    @torch.inference_mode()
    def generate_speaker_prompt_batch(
        self, voice_clone_prompt: VoiceClonePrompt
    ) -> list[torch.Tensor]:
        return [
            self.generate_speaker_prompt(ref_spk_embedding)
            for ref_spk_embedding in voice_clone_prompt["ref_spk_embedding"]
        ]


__all__ = [
    "Qwen3TTSConditionalGenerationBase",
]
