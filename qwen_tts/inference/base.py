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
from typing import Any, Dict, List, Optional, Self, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


class Qwen3TTSModelBase:
    """
    Base class for Qwen3 TTS model wrappers. Provides shared initialization,
    from_pretrained, validation, audio loading, text/tokenization helpers,
    and get_supported_* APIs. Feature-specific generation is in mixins.
    """

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

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

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Instance of cls (e.g. Qwen3TTSVoiceClone, Qwen3TTSVoiceDesign, or Qwen3TTSCustomVoice) containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

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

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
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

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
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
                out[i] = (np.mean(a[0], axis=-1).astype(np.float32), a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    def get_supported_speakers(self) -> Optional[List[str]]:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
