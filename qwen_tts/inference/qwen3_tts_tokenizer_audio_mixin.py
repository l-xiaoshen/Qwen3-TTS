# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import urllib.request
from collections.abc import Sequence
from typing import Optional, Protocol, Union, runtime_checkable
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

AudioInput = Union[
    str,
    np.ndarray,
    Sequence[str],
    Sequence[np.ndarray],
]


@runtime_checkable
class _FeatureExtractorWithSamplingRate(Protocol):
    sampling_rate: int | float


class Qwen3TTSTokenizerAudioMixin:
    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        # Heuristic: no filesystem path separators and long enough.
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
        # Accept both "data:audio/wav;base64,...." and raw base64
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def load_audio(
        self,
        x: str,
        target_sr: int,
    ) -> np.ndarray:
        """
        Load audio from wav path or base64 string, then resample to target_sr.

        Args:
            x (str):
                A wav file path, or a base64 audio string (raw or data URL).
            target_sr (int):
                Target sampling rate.

        Returns:
            np.ndarray:
                1-D float32 waveform at target_sr.
        """
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

        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    def _normalize_audio_inputs(
        self,
        audios: AudioInput,
        sr: Optional[int],
    ) -> list[np.ndarray]:
        """
        Normalize all supported input types into a list of 1-D numpy float32 waveforms
        at `self.feature_extractor.sampling_rate`.

        Args:
            audios (AudioInput):
                - str: wav path OR base64 audio string
                - np.ndarray: raw waveform (sr must be provided)
                - list[str] / list[np.ndarray]
            sr (Optional[int]):
                Sampling rate for raw numpy input. Required if input is np.ndarray or list[np.ndarray].

        Returns:
            list[np.ndarray]:
                List of float32 waveforms resampled to model input SR.
        """
        feature_extractor = getattr(self, "feature_extractor", None)
        if feature_extractor is None or not isinstance(
            feature_extractor, _FeatureExtractorWithSamplingRate
        ):
            raise RuntimeError("Feature extractor is not initialized.")
        target_sr = int(feature_extractor.sampling_rate)

        audio_items: list[str | np.ndarray]
        if isinstance(audios, str):
            audio_items = [audios]
        elif isinstance(audios, np.ndarray):
            audio_items = [audios]
        else:
            audio_items = list(audios)

        if len(audio_items) == 0:
            return []

        if all(isinstance(item, str) for item in audio_items):
            path_items = [item for item in audio_items if isinstance(item, str)]
            load_audio_fn = getattr(self, "load_audio", None)
            if not callable(load_audio_fn):
                raise RuntimeError("Tokenizer does not implement `load_audio`.")
            return [load_audio_fn(path, target_sr=target_sr) for path in path_items]

        waveform_items = [item for item in audio_items if isinstance(item, np.ndarray)]
        if len(waveform_items) != len(audio_items):
            raise TypeError(
                "Mixed input types are not supported. Use all paths/base64 or all numpy arrays."
            )

        if sr is None:
            raise ValueError(
                "For numpy waveform input, you must provide `sr` (original sampling rate)."
            )

        out: list[np.ndarray] = []
        source_sr = int(sr)
        for waveform in waveform_items:
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            wave_fp32 = np.asarray(waveform, dtype=np.float32)
            if source_sr != target_sr:
                wave_fp32 = librosa.resample(
                    y=wave_fp32,
                    orig_sr=source_sr,
                    target_sr=target_sr,
                )
            out.append(np.asarray(wave_fp32, dtype=np.float32))
        return out
