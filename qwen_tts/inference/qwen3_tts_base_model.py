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
import base64
import io
import urllib.request
from collections.abc import Mapping
from typing import Optional, TypeVar, TypedDict, Union
from typing_extensions import Self
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

AudioLike = Union[
    str,  # wav path, URL, base64
    np.ndarray,  # waveform (requires sr)
    tuple[np.ndarray, int],  # (waveform, sr)
]

T = TypeVar("T")
GenerateDefaultValue = bool | int | float
GenerateDefaults = dict[str, GenerateDefaultValue]
GenerateDefaultsInputValue = bool | int | float | str | None
GenerateDefaultsInput = Mapping[str, GenerateDefaultsInputValue]
GenerateExtraArg = bool | int | float | str | None


class GenerateOptions(TypedDict):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_dosample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
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
        model: Qwen3TTSForConditionalGeneration,
        processor: Qwen3TTSProcessor,
        generate_defaults: Optional[GenerateDefaultsInput] = None,
    ):
        self.model = model
        self.processor = processor
        defaults: GenerateDefaults = {}
        if generate_defaults is not None:
            for key, value in generate_defaults.items():
                if isinstance(value, bool):
                    defaults[key] = value
                elif isinstance(value, int):
                    defaults[key] = value
                elif isinstance(value, float):
                    defaults[key] = value
        self.generate_defaults = defaults

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
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
        )

        generate_defaults_raw = model.generate_config
        generate_defaults: GenerateDefaultsInput | None
        if isinstance(generate_defaults_raw, Mapping):
            normalized_defaults: dict[str, GenerateDefaultsInputValue] = {}
            for key, value in generate_defaults_raw.items():
                str_key = str(key)
                if isinstance(value, (bool, int, float, str)) or value is None:
                    normalized_defaults[str_key] = value
            generate_defaults = normalized_defaults
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

    def _supported_languages_set(self) -> Optional[set[str]]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return {str(x).lower() for x in v}
        return None

    def _supported_speakers_set(self) -> Optional[set[str]]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return {str(x).lower() for x in v}
        return None

    def _validate_languages(self, languages: list[str]) -> None:
        """
        Validate that requested languages are supported by the model.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad: list[str] = []
        for lang in languages:
            if lang is None:
                bad.append(str(lang))
                continue
            if str(lang).lower() not in supported:
                bad.append(str(lang))
        if bad:
            raise ValueError(
                f"Unsupported languages: {bad}. Supported: {sorted(supported)}"
            )

    def _validate_speakers(self, speakers: list[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the model.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad: list[str] = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(str(spk))
        if bad:
            raise ValueError(
                f"Unsupported speakers: {bad}. Supported: {sorted(supported)}"
            )

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(
        self, audios: Union[AudioLike, list[AudioLike]]
    ) -> list[tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: list[tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
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

    def _ensure_list(self, x: T | list[T]) -> list[T]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: list[str]) -> list[torch.Tensor]:
        input_ids: list[torch.Tensor] = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: GenerateExtraArg,
    ) -> GenerateOptionsExtended:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.
        """
        hard_defaults = {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 2048,
        }

        def pick_bool(name: str, user_val: Optional[bool]) -> bool:
            if user_val is not None:
                return user_val
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, bool):
                return default_val
            return bool(hard_defaults[name])

        def pick_int(name: str, user_val: Optional[int]) -> int:
            if user_val is not None:
                return user_val
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, int):
                return default_val
            return int(hard_defaults[name])

        def pick_float(name: str, user_val: Optional[float]) -> float:
            if user_val is not None:
                return user_val
            default_val = self.generate_defaults.get(name)
            if isinstance(default_val, (int, float)):
                return float(default_val)
            return float(hard_defaults[name])

        merged = GenerateOptionsExtended(
            do_sample=pick_bool("do_sample", do_sample),
            top_k=pick_int("top_k", top_k),
            top_p=pick_float("top_p", top_p),
            temperature=pick_float("temperature", temperature),
            repetition_penalty=pick_float("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick_bool("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick_int("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick_float("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick_float(
                "subtalker_temperature", subtalker_temperature
            ),
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
            else:
                # Keep backward-compatible behavior: unknown keys are ignored here.
                continue
        return merged

    def get_supported_speakers(self) -> Optional[list[str]]:
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> Optional[list[str]]:
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)

    def _require_speech_tokenizer(self) -> Qwen3TTSTokenizer:
        tokenizer = self.model.speech_tokenizer
        if tokenizer is None:
            raise RuntimeError("Speech tokenizer is not loaded on the model.")
        return tokenizer
