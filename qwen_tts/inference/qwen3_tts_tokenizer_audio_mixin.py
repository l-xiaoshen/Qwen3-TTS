# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import urllib.request
from typing import List, Optional, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

AudioInput = Union[
    str,
    np.ndarray,
    List[str],
    List[np.ndarray],
]


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
    ) -> List[np.ndarray]:
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
            List[np.ndarray]:
                List of float32 waveforms resampled to model input SR.
        """
        target_sr = int(self.feature_extractor.sampling_rate)

        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        if len(audios) == 0:
            return []

        if isinstance(audios[0], str):
            # wav path list or base64 list
            return [self.load_audio(x, target_sr=target_sr) for x in audios]  # type: ignore[arg-type]

        # numpy list
        if sr is None:
            raise ValueError(
                "For numpy waveform input, you must provide `sr` (original sampling rate)."
            )

        out: List[np.ndarray] = []
        for a in audios:  # type: ignore[assignment]
            if not isinstance(a, np.ndarray):
                raise TypeError(
                    "Mixed input types are not supported. Use all paths/base64 or all numpy arrays."
                )
            if a.ndim > 1:
                a = np.mean(a, axis=-1)
            if int(sr) != target_sr:
                a = librosa.resample(
                    y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr
                )
            out.append(a.astype(np.float32))
        return out
