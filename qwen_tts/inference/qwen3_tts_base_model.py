# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import os
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, TypedDict, cast

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor
from typing_extensions import Self

from qwen_tts.core import SpeakerConfiguration, SubTalkerConfiguration

from ..audio_utils import load_audio_to_np_and_sr
from ..core.models import (
    Qwen3TTSConditionalGenerationBase,
    Qwen3TTSConfig,
    Qwen3TTSCustomVoiceForConditionalGeneration,
    Qwen3TTSProcessor,
    Qwen3TTSVoiceCloneForConditionalGeneration,
    Qwen3TTSVoiceDesignForConditionalGeneration,
)
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

AudioLike = (
    str  # wav path, URL, base64
    | np.ndarray  # waveform (requires sr)
    | tuple[np.ndarray, int]  # (waveform, sr)
)


class TTSInputItem(TypedDict):
    text: str
    instruction: str


TTSInput = list[TTSInputItem] | tuple[TTSInputItem, ...]
TTSBatchInput = list[TTSInput] | tuple[TTSInput, ...]


class GenerationDefaults(TypedDict, total=False):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_configuration: SubTalkerConfiguration
    max_new_tokens: int


class ResolvedGenerationOptions(TypedDict):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_configuration: SubTalkerConfiguration
    max_new_tokens: int
    eos_token_id: int | None


class Qwen3TTSBaseModel:
    """
    Shared model wrapper logic for Qwen3 TTS feature-specific classes.

    This class only contains model loading, common preprocessing helpers, and
    shared generation utility methods. Feature APIs are implemented in
    dedicated subclasses.
    """

    _ASSISTANT_PREFIX = "<|im_start|>assistant\n"
    _ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    _USER_PREFIX = "<|im_start|>user\n"
    _MESSAGE_SUFFIX = "<|im_end|>\n"
    _model_class: ClassVar[type[Qwen3TTSConditionalGenerationBase]] = (
        Qwen3TTSConditionalGenerationBase
    )

    def __init__(
        self,
        model: Qwen3TTSConditionalGenerationBase,
        processor: Qwen3TTSProcessor,
        generate_defaults: GenerationDefaults | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.generate_defaults = self._normalize_generate_defaults(generate_defaults)

        model_device: object = getattr(model, "device", None)
        if isinstance(model_device, torch.device):
            self.device = model_device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self._assistant_prefix_ids = self._tokenize_raw_text(self._ASSISTANT_PREFIX)
        self._assistant_suffix_ids = self._tokenize_raw_text(self._ASSISTANT_SUFFIX)
        self._user_prefix_ids = self._tokenize_raw_text(self._USER_PREFIX)
        self._message_suffix_ids = self._tokenize_raw_text(self._MESSAGE_SUFFIX)
        self._validate_prompt_fragments()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
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
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        if isinstance(config, Qwen3TTSConfig):
            resolved_config = config
        else:
            config_source = pretrained_model_name_or_path if config is None else config
            config_loader = cast(Callable[..., object], AutoConfig.from_pretrained)
            config_raw = config_loader(
                config_source,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs,
            )
            if not isinstance(config_raw, Qwen3TTSConfig):
                raise TypeError(
                    f"AutoConfig returned {type(config_raw)}, expected Qwen3TTSConfig."
                )
            resolved_config = config_raw

        if resolved_config.tts_model_type == "voice_design":
            model_cls: type[Qwen3TTSConditionalGenerationBase] = (
                Qwen3TTSVoiceDesignForConditionalGeneration
            )
        elif resolved_config.tts_model_type == "custom_voice":
            model_cls = Qwen3TTSCustomVoiceForConditionalGeneration
        elif resolved_config.tts_model_type == "base":
            model_cls = Qwen3TTSVoiceCloneForConditionalGeneration
        else:
            raise ValueError(
                f"Unsupported tts_model_type: {resolved_config.tts_model_type}"
            )

        model_loader = cast(Callable[..., object], model_cls.from_pretrained)
        model_raw = model_loader(
            pretrained_model_name_or_path,
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
        if not isinstance(model_raw, cls._model_class):
            raise TypeError(
                f"Model loader returned {type(model_raw)}, expected "
                f"{cls._model_class.__name__}."
            )
        model = model_raw

        subfolder = kwargs.get("subfolder")
        if subfolder is not None and not isinstance(subfolder, str):
            raise TypeError("`subfolder` must be a string.")
        processor_kwargs: dict[str, object] = {
            "fix_mistral_regex": True,
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
        }
        if subfolder is not None:
            processor_kwargs["subfolder"] = subfolder
        processor_loader = cast(Callable[..., object], AutoProcessor.from_pretrained)
        processor_raw = processor_loader(
            pretrained_model_name_or_path,
            **processor_kwargs,
        )
        if not isinstance(processor_raw, Qwen3TTSProcessor):
            raise TypeError(
                f"AutoProcessor returned {type(processor_raw)}, "
                "expected Qwen3TTSProcessor."
            )
        processor = processor_raw

        generate_defaults_raw = model.generate_config
        generate_defaults: GenerationDefaults | None
        if isinstance(generate_defaults_raw, Mapping):
            generate_defaults = cls._parse_generate_defaults(generate_defaults_raw)
        else:
            generate_defaults = None
        return cls(
            model=model, processor=processor, generate_defaults=generate_defaults
        )

    def _ensure_model_type(self, expected_type: str, api_name: str) -> None:
        if self.model.tts_model_type != expected_type:
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                f"does not support {api_name}, Please check Model Card or Readme for more details."
            )

    def _supported_languages_set(self) -> set[str] | None:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
                raise TypeError("Model-supported languages must be a sequence.")
            return {str(x).lower() for x in v}
        return None

    def _supported_speakers_set(self) -> set[str] | None:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
                raise TypeError("Model-supported speakers must be a sequence.")
            return {str(x).lower() for x in v}
        return None

    def _speaker_language_bucket(self, speaker: str) -> str | None:
        config = getattr(self.model, "config", None)
        talker_config = getattr(config, "talker_config", None)
        dialect_map = getattr(talker_config, "spk_is_dialect", None) or {}
        language_map = getattr(talker_config, "codec_language_id", None)
        dialect_value = dialect_map.get(speaker.lower(), False)
        if (
            isinstance(dialect_value, str)
            and isinstance(language_map, Mapping)
            and dialect_value in language_map
        ):
            return dialect_value
        return None

    def _validate_speaker_configuration_language_consistency(
        self, speaker: SpeakerConfiguration
    ) -> None:
        active_speakers = [
            speaker_id for speaker_id, weight in speaker.items() if float(weight) != 0.0
        ]
        if len(active_speakers) <= 1:
            return

        first_bucket = self._speaker_language_bucket(active_speakers[0])
        for speaker_id in active_speakers[1:]:
            if self._speaker_language_bucket(speaker_id) != first_bucket:
                raise ValueError(
                    "All speakers in `speaker` must resolve to the same language bucket."
                )

    def _validate_languages(self, languages: list[str]) -> None:
        """
        Validate that requested languages are supported by the model.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad: list[str] = []
        for lang in languages:
            if str(lang).lower() not in supported:
                bad.append(str(lang))
        if bad:
            raise ValueError(
                f"Unsupported languages: {bad}. Supported: {sorted(supported)}"
            )

    def _validate_speaker_configuration(self, speaker: SpeakerConfiguration) -> None:
        """
        Validate that requested speaker configuration is supported by the model.
        """
        supported = self._supported_speakers_set()
        bad: list[str] = []
        for speaker_id in speaker:
            if (
                supported is not None
                and speaker_id != ""
                and speaker_id.lower() not in supported
            ):
                bad.append(speaker_id)

        if bad:
            supported_list = sorted(supported) if supported is not None else []
            raise ValueError(
                f"Unsupported speakers: {bad}. Supported: {supported_list}"
            )
        self._validate_speaker_configuration_language_consistency(speaker)

    def _validate_speaker_configurations(
        self,
        speaker_configurations: list[SpeakerConfiguration]
        | tuple[SpeakerConfiguration, ...],
        batch_size: int,
    ) -> None:
        if len(speaker_configurations) != batch_size:
            raise ValueError(
                f"Batch size mismatch: text={batch_size}, speaker={len(speaker_configurations)}"
            )
        for speaker_configuration in speaker_configurations:
            self._validate_speaker_configuration(speaker_configuration)

    def _validate_speakers(self, speakers: list[str]) -> None:
        """
        Validate that requested speakers are supported by the model.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad: list[str] = []
        for spk in speakers:
            if spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(str(spk))
        if bad:
            raise ValueError(
                f"Unsupported speakers: {bad}. Supported: {sorted(supported)}"
            )

    def _normalize_audio_inputs(
        self, audios: list[AudioLike] | tuple[AudioLike, ...]
    ) -> list[tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).
        """
        out: list[tuple[np.ndarray, int]] = []
        for a in audios:
            if isinstance(a, str):
                out.append(load_audio_to_np_and_sr(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                mono = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (mono, a[1])
        return out

    def _tokenize_raw_text(self, text: str) -> torch.Tensor:
        input_data_raw: object = self.processor(
            text=text, return_tensors="pt", padding=True
        )
        if not isinstance(input_data_raw, BatchFeature):
            raise TypeError("Processor output must be a BatchFeature.")
        input_ids_raw: object = input_data_raw.get("input_ids")
        if not isinstance(input_ids_raw, torch.Tensor):
            raise TypeError("Processor output `input_ids` must be a tensor.")
        input_id = input_ids_raw.to(self.device)
        return input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

    def _validate_prompt_fragments(self) -> None:
        fragment_lengths = {
            "assistant prefix": self._assistant_prefix_ids.shape[1],
            "assistant suffix": self._assistant_suffix_ids.shape[1],
            "user prefix": self._user_prefix_ids.shape[1],
            "message suffix": self._message_suffix_ids.shape[1],
        }
        expected_lengths = {
            "assistant prefix": 3,
            "assistant suffix": 5,
            "user prefix": 3,
            "message suffix": 2,
        }
        if fragment_lengths != expected_lengths:
            raise ValueError(
                "The text tokenizer produced an unsupported Qwen TTS prompt layout: "
                f"{fragment_lengths}. Expected {expected_lengths}."
            )

    def _tokenize_framed_text(
        self,
        text: str,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_ids = self._tokenize_raw_text(text)
        return torch.cat((prefix_ids, text_ids, suffix_ids), dim=1)

    def _tokenize_assistant_input(self, text: str) -> torch.Tensor:
        return self._tokenize_framed_text(
            text,
            self._assistant_prefix_ids,
            self._assistant_suffix_ids,
        )

    def _tokenize_instruction(self, instruction: str) -> torch.Tensor | None:
        if instruction == "":
            return None
        return self._tokenize_framed_text(
            instruction,
            self._user_prefix_ids,
            self._message_suffix_ids,
        )

    def _tokenize_ref_text(self, ref_text: str) -> torch.Tensor | None:
        if ref_text == "":
            return None
        return self._tokenize_framed_text(
            ref_text,
            self._assistant_prefix_ids,
            self._message_suffix_ids,
        )

    def _normalize_tts_input(
        self,
        tts_input: TTSInput,
        input_name: str = "tts_input",
    ) -> list[TTSInputItem]:
        if len(tts_input) == 0:
            raise ValueError(f"`{input_name}` must contain at least one chunk.")

        chunks: list[TTSInputItem] = []
        for index, chunk in enumerate(tts_input):
            text = chunk["text"]
            instruction = chunk["instruction"]
            if text.strip() == "":
                raise ValueError(f"`{input_name}[{index}].text` must be non-empty.")
            chunks.append(TTSInputItem(text=text, instruction=instruction))
        return chunks

    def _normalize_tts_batch_input(
        self, tts_input: TTSBatchInput
    ) -> list[list[TTSInputItem]]:
        if len(tts_input) == 0:
            raise ValueError("`tts_input` must contain at least one input.")

        return [
            self._normalize_tts_input(
                item,
                input_name=f"tts_input[{index}]",
            )
            for index, item in enumerate(tts_input)
        ]

    def _tokenize_tts_chunks(
        self, chunks: Sequence[TTSInputItem]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        return (
            [self._tokenize_assistant_input(chunk["text"]) for chunk in chunks],
            [self._tokenize_instruction(chunk["instruction"]) for chunk in chunks],
        )

    def _normalize_language_value(self, language: str) -> str:
        return "Auto" if language == "" else language

    def _normalize_language_values(
        self, languages: list[str] | tuple[str, ...], batch_size: int
    ) -> list[str]:
        if len(languages) == 0:
            return ["Auto"] * batch_size
        return list(languages)

    def _tokenize_ref_texts(
        self, ref_texts: Sequence[str]
    ) -> list[torch.Tensor | None]:
        return [self._tokenize_ref_text(ref_text) for ref_text in ref_texts]

    @staticmethod
    def _build_subtalker_configuration(
        subtalker_configuration_items: list[tuple[str, object]],
    ) -> SubTalkerConfiguration:
        normalized: SubTalkerConfiguration = {}
        for key, value in subtalker_configuration_items:
            if key == "do_sample":
                if not isinstance(value, bool):
                    raise TypeError(
                        "`subtalker_configuration['do_sample']` must be a boolean."
                    )
                normalized["do_sample"] = value
            elif key == "top_k":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(
                        "`subtalker_configuration['top_k']` must be an integer."
                    )
                normalized["top_k"] = value
            elif key == "top_p":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        "`subtalker_configuration['top_p']` must be numeric."
                    )
                normalized["top_p"] = float(value)
            elif key == "temperature":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        "`subtalker_configuration['temperature']` must be numeric."
                    )
                normalized["temperature"] = float(value)
            else:
                raise ValueError(
                    "Unsupported `subtalker_configuration` key: "
                    f"{key!r}. Supported keys are "
                    "'do_sample', 'top_k', 'top_p', and 'temperature'."
                )
        return normalized

    @classmethod
    def _parse_subtalker_configuration(
        cls,
        subtalker_configuration: Mapping[object, object] | None,
    ) -> SubTalkerConfiguration:
        if subtalker_configuration is None:
            return SubTalkerConfiguration()
        if not isinstance(subtalker_configuration, Mapping):
            raise TypeError("`subtalker_configuration` must be a mapping.")
        return cls._build_subtalker_configuration(
            [(str(key), value) for key, value in subtalker_configuration.items()]
        )

    @classmethod
    def _parse_generate_defaults(
        cls,
        generate_defaults: Mapping[str, object] | None,
    ) -> GenerationDefaults | None:
        if generate_defaults is None:
            return None

        parsed_generate_defaults = GenerationDefaults()

        do_sample = generate_defaults.get("do_sample")
        if do_sample is not None:
            if not isinstance(do_sample, bool):
                raise TypeError("`generate_defaults['do_sample']` must be a boolean.")
            parsed_generate_defaults["do_sample"] = do_sample

        for key in ("top_k", "max_new_tokens"):
            value = generate_defaults.get(key)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"`generate_defaults[{key!r}]` must be an integer.")
            if key == "top_k":
                parsed_generate_defaults["top_k"] = value
            else:
                parsed_generate_defaults["max_new_tokens"] = value

        for key in ("top_p", "temperature", "repetition_penalty"):
            value = generate_defaults.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"`generate_defaults[{key!r}]` must be numeric.")
            if key == "top_p":
                parsed_generate_defaults["top_p"] = float(value)
            elif key == "temperature":
                parsed_generate_defaults["temperature"] = float(value)
            else:
                parsed_generate_defaults["repetition_penalty"] = float(value)

        subtalker_configuration = generate_defaults.get("subtalker_configuration")
        if subtalker_configuration is not None:
            if not isinstance(subtalker_configuration, Mapping):
                raise TypeError(
                    "`generate_defaults['subtalker_configuration']` must be a mapping."
                )
            parsed_generate_defaults["subtalker_configuration"] = (
                cls._parse_subtalker_configuration(
                    cast(Mapping[object, object], subtalker_configuration)
                )
            )
        return parsed_generate_defaults

    def _normalize_generate_defaults(
        self,
        generate_defaults: GenerationDefaults | None,
    ) -> GenerationDefaults:
        normalized_generate_defaults = GenerationDefaults()
        if generate_defaults is None:
            return normalized_generate_defaults

        normalized_generate_defaults.update(generate_defaults)
        subtalker_configuration = generate_defaults.get("subtalker_configuration")
        if subtalker_configuration is not None:
            normalized_generate_defaults["subtalker_configuration"] = (
                self._normalize_subtalker_configuration(subtalker_configuration)
            )
        return normalized_generate_defaults

    def _normalize_subtalker_configuration(
        self,
        subtalker_configuration: SubTalkerConfiguration | None,
    ) -> SubTalkerConfiguration:
        if subtalker_configuration is None:
            return SubTalkerConfiguration()
        normalized = SubTalkerConfiguration()
        normalized.update(subtalker_configuration)
        return normalized

    def _decode_talker_codes_batch(
        self, talker_codes_list: Sequence[torch.Tensor]
    ) -> tuple[list[np.ndarray], int]:
        speech_tokenizer = self._require_speech_tokenizer()
        wavs, fs = speech_tokenizer.decode(
            [{"audio_codes": codes} for codes in talker_codes_list]
        )
        return wavs, fs

    def _decode_talker_turns(
        self,
        talker_codes_list: Sequence[torch.Tensor],
        prefix_code: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], int]:
        if len(talker_codes_list) == 0:
            raise ValueError("`talker_codes_list` must contain at least one turn.")

        first_code = talker_codes_list[0]
        code_parts: list[torch.Tensor] = []
        history_length = 0
        if prefix_code is not None:
            history_length = int(prefix_code.shape[0])
            code_parts.append(
                prefix_code.to(device=first_code.device, dtype=first_code.dtype)
            )

        samples_per_code = self._require_speech_tokenizer().get_decode_upsample_rate()
        turn_wavs: list[np.ndarray] = []
        sample_rate: int | None = None
        for talker_codes in talker_codes_list:
            code_parts.append(talker_codes)
            cumulative_codes = torch.cat(code_parts, dim=0)
            decoded, fs = self._decode_talker_codes_batch([cumulative_codes])
            if len(decoded) != 1:
                raise RuntimeError("Turn decoding produced an unexpected output count.")
            if sample_rate is not None and fs != sample_rate:
                raise RuntimeError("Decoded sample rates differ across turns.")
            sample_rate = fs

            expected_length = int(cumulative_codes.shape[0]) * samples_per_code
            waveform = decoded[0]
            if waveform.shape[0] < expected_length:
                raise RuntimeError(
                    "Decoded waveform is shorter than its codec sequence requires."
                )

            offset = history_length * samples_per_code
            turn_wavs.append(waveform[offset:expected_length].copy())
            history_length = int(cumulative_codes.shape[0])
        if sample_rate is None:
            raise RuntimeError("Turn decoding produced no outputs.")
        return turn_wavs, sample_rate

    def _resolve_generation_options(
        self,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> ResolvedGenerationOptions:
        """
        Resolve user overrides against defaults from `generation_config.json`.
        """
        normalized_subtalker_configuration = self._normalize_subtalker_configuration(
            subtalker_configuration
        )
        generate_default_subtalker_configuration_value = self.generate_defaults.get(
            "subtalker_configuration"
        )
        if generate_default_subtalker_configuration_value is None:
            generate_default_subtalker_configuration = SubTalkerConfiguration()
        else:
            generate_default_subtalker_configuration = (
                generate_default_subtalker_configuration_value
            )

        resolved_subtalker_configuration = SubTalkerConfiguration(
            do_sample=normalized_subtalker_configuration.get(
                "do_sample",
                generate_default_subtalker_configuration.get("do_sample", True),
            ),
            top_k=normalized_subtalker_configuration.get(
                "top_k",
                generate_default_subtalker_configuration.get("top_k", 50),
            ),
            top_p=normalized_subtalker_configuration.get(
                "top_p",
                generate_default_subtalker_configuration.get("top_p", 1.0),
            ),
            temperature=normalized_subtalker_configuration.get(
                "temperature",
                generate_default_subtalker_configuration.get("temperature", 0.9),
            ),
        )

        return ResolvedGenerationOptions(
            do_sample=(
                do_sample
                if do_sample is not None
                else self.generate_defaults.get("do_sample", True)
            ),
            top_k=(
                top_k if top_k is not None else self.generate_defaults.get("top_k", 50)
            ),
            top_p=(
                top_p if top_p is not None else self.generate_defaults.get("top_p", 1.0)
            ),
            temperature=(
                temperature
                if temperature is not None
                else self.generate_defaults.get("temperature", 0.9)
            ),
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self.generate_defaults.get("repetition_penalty", 1.05)
            ),
            subtalker_configuration=resolved_subtalker_configuration,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self.generate_defaults.get("max_new_tokens", 2048)
            ),
            eos_token_id=eos_token_id,
        )

    def get_supported_speakers(self) -> list[str] | None:
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> list[str] | None:
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)

    def _require_speech_tokenizer(self) -> Qwen3TTSTokenizer:
        tokenizer = self.model.speech_tokenizer
        if tokenizer is None:
            raise RuntimeError("Speech tokenizer is not loaded on the model.")
        return tokenizer
