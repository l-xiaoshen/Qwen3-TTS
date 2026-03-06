# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

from ..audio_utils import load_audio_to_np_and_sr, resample_audio

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
        audio, source_sr = load_audio_to_np_and_sr(x)
        return resample_audio(audio, source_sr=source_sr, target_sr=target_sr)

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
            out.append(
                resample_audio(waveform, source_sr=source_sr, target_sr=target_sr)
            )
        return out
