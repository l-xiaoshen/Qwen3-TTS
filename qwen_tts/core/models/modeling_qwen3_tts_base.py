"""Base conditional-generation model for Qwen3 TTS."""

import json
import os
from collections.abc import Callable, Sequence
from typing import cast

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.utils import logging
from transformers.utils.hub import cached_file
from typing_extensions import Self, override

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
    model_name_or_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None,
    allow_patterns: Sequence[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
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
    if not allow_patterns:
        raise ValueError("`allow_patterns` must not be empty.")
    local_only = local_files_only or huggingface_hub.constants.HF_HUB_OFFLINE
    repo_id = os.fspath(model_name_or_path)
    resolved_cache_dir = os.fspath(cache_dir) if cache_dir is not None else None

    return snapshot_download(
        repo_id,
        allow_patterns=list(allow_patterns),
        ignore_patterns=ignore_patterns,
        cache_dir=resolved_cache_dir,
        revision=revision,
        force_download=force_download,
        local_files_only=local_only,
        token=token,
    )


def _parse_generate_config(value: object) -> dict[str, GenerateConfigValue]:
    if not isinstance(value, dict):
        raise TypeError("Generation config must be a JSON object.")
    return cast(dict[str, GenerateConfigValue], value)


class Qwen3TTSConditionalGenerationBase(
    Qwen3TTSGenerationMixin, Qwen3TTSPreTrainedModel
):
    config_class = Qwen3TTSConfig

    def __init__(self, config: Qwen3TTSConfig) -> None:
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
        for language_id in supported_languages:
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
    @override
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: object,
        config: Qwen3TTSConfig | str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs: object,
    ) -> Self:
        # Hotfix to enable passing the correct attn implementation which is stored in the config but not in kwargs
        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if (
            requested_attn_implementation is None
            and isinstance(config, Qwen3TTSConfig)
            and config._attn_implementation
        ):
            requested_attn_implementation = config._attn_implementation

        model_loader = cast(Callable[..., Self], super().from_pretrained)
        model = model_loader(
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
        subfolder = kwargs.get("subfolder")
        if subfolder is not None and not isinstance(subfolder, str):
            raise TypeError("`subfolder` must be a string.")
        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            speech_tokenizer_pattern = "speech_tokenizer/*"
            if subfolder:
                speech_tokenizer_pattern = (
                    f"{subfolder.rstrip('/')}/{speech_tokenizer_pattern}"
                )
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                allow_patterns=[speech_tokenizer_pattern],
                revision=revision,
                force_download=force_download,
                token=token,
            )
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if speech_tokenizer_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{speech_tokenizer_path} not exists"""
            )
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        component_kwargs = dict(kwargs)
        component_kwargs.pop("subfolder", None)
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            *model_args,
            **component_kwargs,
        )
        model.load_speech_tokenizer(speech_tokenizer)

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if generate_config_path is None:
            raise ValueError(
                f"{pretrained_model_name_or_path}/generation_config.json not exists"
            )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            generate_config = _parse_generate_config(json.load(f))
        model.load_generate_config(generate_config)

        return model

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            raise ValueError("Only 24 kHz audio is supported.")
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
        speaker_embeddings = cast(
            torch.Tensor, speaker_encoder(mels.to(self.device).to(self.dtype))
        )
        speaker_embedding = speaker_embeddings[0]
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
