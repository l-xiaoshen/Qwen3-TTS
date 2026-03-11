# coding=utf-8
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, Union
from typing_extensions import Self

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from qwen_tts.core import SpeakerConfiguration, SubTalkerConfiguration

from ..audio_utils import load_audio_to_np_and_sr
from ..core.models import (
    Qwen3TTSConfig,
    Qwen3TTSConditionalGenerationBase,
    Qwen3TTSCustomVoiceForConditionalGeneration,
    Qwen3TTSProcessor,
    Qwen3TTSVoiceCloneForConditionalGeneration,
    Qwen3TTSVoiceDesignForConditionalGeneration,
)
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

AudioLike = Union[
    str,  # wav path, URL, base64
    np.ndarray,  # waveform (requires sr)
    tuple[np.ndarray, int],  # (waveform, sr)
]

GenerateDefaultValue = bool | int | float | SubTalkerConfiguration
GenerateDefaults = dict[str, GenerateDefaultValue]
GenerateExtraArg = bool | int | float | str


class TTSInputPart(TypedDict):
    instruction: str
    text: str


TTSInput = list[TTSInputPart]
TTSInputs = list[TTSInput]


@dataclass(frozen=True)
class TokenizedTTSInputPart:
    instruction_id: torch.Tensor | None
    input_role_id: torch.Tensor
    input_text_id: torch.Tensor


class GenerateOptions(TypedDict):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_configuration: SubTalkerConfiguration
    max_new_tokens: int


class GenerateOptionsExtended(GenerateOptions, total=False):
    eos_token_id: int
    output_hidden_states: bool
    return_dict_in_generate: bool


class Qwen3TTSBaseModel:
    """
    Shared model wrapper logic for Qwen3 TTS feature-specific classes.

    This class only contains model loading, common preprocessing helpers, and
    shared generation utility methods. Feature APIs are implemented in
    dedicated subclasses.
    """

    def __init__(
        self,
        model: Qwen3TTSConditionalGenerationBase,
        processor: Qwen3TTSProcessor,
        generate_defaults: GenerateDefaults | None = None,
    ):
        self.model = model
        self.processor = processor
        self.generate_defaults = self._normalize_generate_defaults(generate_defaults)

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> Self:
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(config, Qwen3TTSConfig):
            raise TypeError(
                f"AutoConfig returned {type(config)}, expected Qwen3TTSConfig."
            )

        if config.tts_model_type == "voice_design":
            model_cls = Qwen3TTSVoiceDesignForConditionalGeneration
        elif config.tts_model_type == "custom_voice":
            model_cls = Qwen3TTSCustomVoiceForConditionalGeneration
        elif config.tts_model_type == "base":
            model_cls = Qwen3TTSVoiceCloneForConditionalGeneration
        else:
            raise ValueError(f"Unsupported tts_model_type: {config.tts_model_type}")

        model = model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSConditionalGenerationBase):
            raise TypeError(
                f"Model loader returned {type(model)}, expected Qwen3TTSConditionalGenerationBase subclass."
            )

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
        )

        generate_defaults_raw = model.generate_config
        generate_defaults: GenerateDefaults | None
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
            return {str(x).lower() for x in v}
        return None

    def _supported_speakers_set(self) -> set[str] | None:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
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
        if not isinstance(speaker, dict):
            raise TypeError("`speaker` must be a dict mapping speaker ids to weights.")

        supported = self._supported_speakers_set()
        bad: list[str] = []
        for speaker_id, weight in speaker.items():
            if not isinstance(speaker_id, str):
                raise TypeError("`speaker` keys must be strings.")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise TypeError("`speaker` values must be numeric weights.")
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
        speaker_configurations: Sequence[SpeakerConfiguration],
        batch_size: int,
    ) -> None:
        if not isinstance(speaker_configurations, Sequence) or isinstance(
            speaker_configurations, (str, bytes)
        ):
            raise TypeError(
                "`speaker` must be a sequence of speaker configuration dictionaries."
            )
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
        self, audios: list[AudioLike]
    ) -> list[tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).
        """
        if not isinstance(audios, list):
            raise TypeError("`audios` must be a list of audio inputs.")
        items = audios

        out: list[tuple[np.ndarray, int]] = []
        for a in items:
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

    @staticmethod
    def _assistant_role_prefix() -> str:
        return "<|im_start|>assistant\n"

    @staticmethod
    def _user_role_prefix() -> str:
        return "<|im_start|>user\n"

    @staticmethod
    def _turn_suffix() -> str:
        return "<|im_end|>\n"

    @staticmethod
    def _normalize_tts_input_part(tts_input_part: object, index: int) -> TTSInputPart:
        if not isinstance(tts_input_part, Mapping):
            raise TypeError(
                f"`tts_input[{index}]` must be a mapping with `instruction` and `text`."
            )
        normalized_tts_input_part: dict[str, object] = {}
        for key, value in tts_input_part.items():
            if not isinstance(key, str):
                raise TypeError(f"`tts_input[{index}]` keys must be strings.")
            normalized_tts_input_part[key] = value

        instruction = normalized_tts_input_part.get("instruction")
        text = normalized_tts_input_part.get("text")
        if not isinstance(instruction, str):
            raise TypeError(f"`tts_input[{index}].instruction` must be a string.")
        if not isinstance(text, str):
            raise TypeError(f"`tts_input[{index}].text` must be a string.")
        if text.strip() == "":
            raise ValueError(f"`tts_input[{index}].text` must be non-empty.")
        return TTSInputPart(instruction=instruction, text=text)

    @classmethod
    def _normalize_tts_input(cls, tts_input: object) -> TTSInput:
        if not isinstance(tts_input, Sequence) or isinstance(tts_input, (str, bytes)):
            raise TypeError("`tts_input` must be a sequence of TTSInputPart items.")

        normalized_tts_input: TTSInput = []
        for index, tts_input_part in enumerate(tts_input):
            normalized_tts_input.append(
                cls._normalize_tts_input_part(tts_input_part, index)
            )
        if len(normalized_tts_input) == 0:
            raise ValueError("`tts_input` must contain at least one part.")
        return normalized_tts_input

    @classmethod
    def _normalize_tts_inputs(cls, tts_inputs: object) -> TTSInputs:
        if not isinstance(tts_inputs, Sequence) or isinstance(tts_inputs, (str, bytes)):
            raise TypeError("`tts_inputs` must be a sequence of TTSInput requests.")

        normalized_tts_inputs: TTSInputs = []
        for index, tts_input in enumerate(tts_inputs):
            try:
                normalized_tts_inputs.append(cls._normalize_tts_input(tts_input))
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"`tts_inputs[{index}]`: {exc}") from exc
        if len(normalized_tts_inputs) == 0:
            raise ValueError("`tts_inputs` must contain at least one request.")
        return normalized_tts_inputs

    def _tokenize_text(
        self, text: str, return_offsets_mapping: bool = False
    ) -> Mapping[str, object]:
        return self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=return_offsets_mapping,
        )

    def _extract_input_ids(self, input_data: Mapping[str, object]) -> torch.Tensor:
        input_ids = input_data.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("Tokenizer output does not contain tensor `input_ids`.")
        input_ids = input_ids.to(self.device)
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    @staticmethod
    def _extract_offset_mapping(input_data: Mapping[str, object]) -> torch.Tensor:
        offset_mapping = input_data.get("offset_mapping")
        if not isinstance(offset_mapping, torch.Tensor):
            raise RuntimeError(
                "Tokenizer output does not contain tensor `offset_mapping`."
            )
        return offset_mapping

    @staticmethod
    def _locate_token_span(
        offset_mapping: torch.Tensor, content_start_char: int, content_end_char: int
    ) -> tuple[int, int]:
        if offset_mapping.dim() == 3:
            offsets = offset_mapping[0]
        elif offset_mapping.dim() == 2:
            offsets = offset_mapping
        else:
            raise RuntimeError("Unexpected `offset_mapping` shape.")

        token_start: int | None = None
        token_end: int | None = None
        for index, (token_start_char, token_end_char) in enumerate(offsets.tolist()):
            if token_start_char >= token_end_char:
                continue
            if (
                token_end_char <= content_start_char
                or token_start_char >= content_end_char
            ):
                continue
            if token_start is None:
                token_start = index
            token_end = index + 1

        if token_start is None or token_end is None:
            raise RuntimeError("Unable to locate content token span in prompt.")
        return token_start, token_end

    def _tokenize_assistant_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        role_prefix = self._assistant_role_prefix()
        rendered_prompt = f"{role_prefix}{text}"
        input_data = self._tokenize_text(
            rendered_prompt,
            return_offsets_mapping=True,
        )
        input_ids = self._extract_input_ids(input_data)
        token_start, token_end = self._locate_token_span(
            self._extract_offset_mapping(input_data),
            content_start_char=len(role_prefix),
            content_end_char=len(rendered_prompt),
        )
        return input_ids[:, :token_start], input_ids[:, token_start:token_end]

    def _tokenize_instruction_text(self, instruction: str) -> torch.Tensor | None:
        if instruction == "":
            return None
        rendered_prompt = (
            f"{self._user_role_prefix()}{instruction}{self._turn_suffix()}"
        )
        return self._extract_input_ids(self._tokenize_text(rendered_prompt))

    def _tokenize_reference_text(self, ref_text: str) -> torch.Tensor | None:
        if ref_text == "":
            return None
        _, ref_text_ids = self._tokenize_assistant_text(ref_text)
        return ref_text_ids

    def _tokenize_tts_input_part(
        self, tts_input_part: TTSInputPart
    ) -> TokenizedTTSInputPart:
        input_role_id, input_text_id = self._tokenize_assistant_text(
            tts_input_part["text"]
        )
        return TokenizedTTSInputPart(
            instruction_id=self._tokenize_instruction_text(
                tts_input_part["instruction"]
            ),
            input_role_id=input_role_id,
            input_text_id=input_text_id,
        )

    def _tokenize_tts_input(self, tts_input: TTSInput) -> list[TokenizedTTSInputPart]:
        return [
            self._tokenize_tts_input_part(tts_input_part)
            for tts_input_part in tts_input
        ]

    @staticmethod
    def _concat_text_context(
        existing_text_id: torch.Tensor | None, new_text_id: torch.Tensor
    ) -> torch.Tensor:
        if existing_text_id is None:
            return new_text_id
        return torch.cat(
            [existing_text_id, new_text_id.to(existing_text_id.device)],
            dim=1,
        )

    @staticmethod
    def _concat_code_context(
        existing_code: torch.Tensor | None, new_code: torch.Tensor
    ) -> torch.Tensor:
        if existing_code is None:
            return new_code
        return torch.cat(
            [existing_code, new_code.to(existing_code.device)],
            dim=0,
        )

    def _normalize_language_value(self, language: str) -> str:
        if not isinstance(language, str):
            raise TypeError("`language` must be a string.")
        return "Auto" if language == "" else language

    def _normalize_language_values(
        self, languages: Sequence[str], batch_size: int
    ) -> list[str]:
        if not isinstance(languages, Sequence) or isinstance(languages, (str, bytes)):
            raise TypeError("`language` must be a sequence of strings.")
        if len(languages) == 0:
            return ["Auto"] * batch_size

        normalized_languages: list[str] = []
        for item in languages:
            if not isinstance(item, str):
                raise TypeError("`language` list items must be strings.")
            normalized_languages.append(item)
        return normalized_languages

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

    @staticmethod
    def _coerce_subtalker_mapping(
        value: object, field_name: str
    ) -> dict[object, object]:
        if not isinstance(value, Mapping):
            raise TypeError(f"`{field_name}` must be a mapping.")
        normalized_mapping: dict[object, object] = {}
        for key, item in value.items():
            normalized_mapping[key] = item
        return normalized_mapping

    @classmethod
    def _parse_subtalker_configuration(
        cls,
        subtalker_configuration: object | None,
    ) -> SubTalkerConfiguration:
        if subtalker_configuration is None:
            return SubTalkerConfiguration()
        subtalker_configuration_mapping = cls._coerce_subtalker_mapping(
            subtalker_configuration, "subtalker_configuration"
        )
        return cls._build_subtalker_configuration(
            [
                (str(key), value)
                for key, value in subtalker_configuration_mapping.items()
            ]
        )

    @classmethod
    def _parse_generate_defaults(
        cls,
        generate_defaults: Mapping[object, object] | None,
    ) -> GenerateDefaults | None:
        if generate_defaults is None:
            return None

        parsed_generate_defaults: GenerateDefaults = {}
        for key, value in generate_defaults.items():
            str_key = str(key)
            if isinstance(value, bool):
                parsed_generate_defaults[str_key] = value
            elif isinstance(value, int):
                parsed_generate_defaults[str_key] = value
            elif isinstance(value, float):
                parsed_generate_defaults[str_key] = value
            elif str_key == "subtalker_configuration":
                if value is None:
                    continue
                parsed_generate_defaults[str_key] = cls._parse_subtalker_configuration(
                    value
                )
        return parsed_generate_defaults

    def _normalize_generate_defaults(
        self,
        generate_defaults: GenerateDefaults | None,
    ) -> GenerateDefaults:
        normalized_generate_defaults: GenerateDefaults = {}
        if generate_defaults is None:
            return normalized_generate_defaults

        for key, value in generate_defaults.items():
            if key == "subtalker_configuration":
                if not isinstance(value, dict):
                    raise TypeError(
                        "`generate_defaults['subtalker_configuration']` must be a SubTalkerConfiguration."
                    )
                normalized_generate_defaults[key] = (
                    self._normalize_subtalker_configuration(value)
                )
            elif isinstance(value, bool):
                normalized_generate_defaults[key] = value
            elif isinstance(value, int):
                normalized_generate_defaults[key] = value
            elif isinstance(value, float):
                normalized_generate_defaults[key] = value
            else:
                raise TypeError(
                    "`generate_defaults` values must be bool, int, float, or "
                    "`SubTalkerConfiguration`."
                )
        return normalized_generate_defaults

    def _normalize_subtalker_configuration(
        self,
        subtalker_configuration: object | None,
    ) -> SubTalkerConfiguration:
        if subtalker_configuration is None:
            return SubTalkerConfiguration()
        subtalker_configuration_mapping = self._coerce_subtalker_mapping(
            subtalker_configuration, "subtalker_configuration"
        )
        return self._build_subtalker_configuration(
            [
                (str(key), value)
                for key, value in subtalker_configuration_mapping.items()
            ]
        )

    def _decode_talker_codes_batch(
        self, talker_codes_list: Sequence[torch.Tensor]
    ) -> tuple[list[np.ndarray], int]:
        speech_tokenizer = self._require_speech_tokenizer()
        wavs, fs = speech_tokenizer.decode(
            [{"audio_codes": codes} for codes in talker_codes_list]
        )
        return wavs, fs

    def _merge_generate_kwargs(
        self,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int = 2048,
        **kwargs: GenerateExtraArg,
    ) -> GenerateOptionsExtended:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.
        """
        normalized_subtalker_configuration = self._normalize_subtalker_configuration(
            subtalker_configuration
        )

        unsupported_kwargs = sorted(
            key
            for key in kwargs
            if key
            not in {
                "eos_token_id",
                "output_hidden_states",
                "return_dict_in_generate",
            }
        )
        if len(unsupported_kwargs) != 0:
            raise TypeError(f"Unsupported generation kwargs: {unsupported_kwargs}")

        hard_defaults = {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "max_new_tokens": 2048,
        }
        hard_subtalker_defaults = SubTalkerConfiguration(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
        )
        generate_default_subtalker_configuration_value = self.generate_defaults.get(
            "subtalker_configuration"
        )
        generate_default_subtalker_configuration = (
            self._normalize_subtalker_configuration(
                generate_default_subtalker_configuration_value
            )
        )

        def pick_bool(name: str, user_val: bool) -> bool:
            hard_default = bool(hard_defaults[name])
            if user_val != hard_default:
                return user_val
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, bool):
                return default_val
            return hard_default

        def pick_int(name: str, user_val: int) -> int:
            hard_default = int(hard_defaults[name])
            if user_val != hard_default:
                return user_val
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, int):
                return default_val
            return hard_default

        def pick_float(name: str, user_val: float) -> float:
            hard_default = float(hard_defaults[name])
            if user_val != hard_default:
                return float(user_val)
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, (int, float)):
                return float(default_val)
            return hard_default

        def pick_subtalker_bool(
            user_val: bool, default_val: object, hard_default: bool
        ) -> bool:
            if user_val != hard_default:
                return user_val
            if isinstance(default_val, bool):
                return default_val
            return hard_default

        def pick_subtalker_int(
            user_val: int, default_val: object, hard_default: int
        ) -> int:
            if user_val != hard_default:
                return user_val
            if isinstance(default_val, int):
                return default_val
            return hard_default

        def pick_subtalker_float(
            user_val: float, default_val: object, hard_default: float
        ) -> float:
            if user_val != hard_default:
                return float(user_val)
            if isinstance(default_val, (int, float)):
                return float(default_val)
            return hard_default

        resolved_subtalker_configuration = SubTalkerConfiguration(
            do_sample=pick_subtalker_bool(
                normalized_subtalker_configuration.get("do_sample", True),
                generate_default_subtalker_configuration.get("do_sample"),
                bool(hard_subtalker_defaults["do_sample"]),
            ),
            top_k=pick_subtalker_int(
                normalized_subtalker_configuration.get("top_k", 50),
                generate_default_subtalker_configuration.get("top_k"),
                int(hard_subtalker_defaults["top_k"]),
            ),
            top_p=pick_subtalker_float(
                float(normalized_subtalker_configuration.get("top_p", 1.0)),
                generate_default_subtalker_configuration.get("top_p"),
                float(hard_subtalker_defaults["top_p"]),
            ),
            temperature=pick_subtalker_float(
                float(normalized_subtalker_configuration.get("temperature", 0.9)),
                generate_default_subtalker_configuration.get("temperature"),
                float(hard_subtalker_defaults["temperature"]),
            ),
        )

        merged = GenerateOptionsExtended(
            do_sample=pick_bool("do_sample", do_sample),
            top_k=pick_int("top_k", top_k),
            top_p=pick_float("top_p", top_p),
            temperature=pick_float("temperature", temperature),
            repetition_penalty=pick_float("repetition_penalty", repetition_penalty),
            subtalker_configuration=resolved_subtalker_configuration,
            max_new_tokens=pick_int("max_new_tokens", max_new_tokens),
        )
        for key, value in kwargs.items():
            if key == "eos_token_id":
                if isinstance(value, int):
                    merged["eos_token_id"] = value
            elif key == "output_hidden_states":
                if isinstance(value, bool):
                    merged["output_hidden_states"] = value
            elif key == "return_dict_in_generate":
                if isinstance(value, bool):
                    merged["return_dict_in_generate"] = value
        return merged

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
