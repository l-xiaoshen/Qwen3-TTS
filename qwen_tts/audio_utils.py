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
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf


def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def is_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def decode_base64_to_wav_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_to_np_and_sr(x: str) -> tuple[np.ndarray, int]:
    if is_url(x):
        with urllib.request.urlopen(x) as response:
            audio_bytes = response.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        wav_bytes = decode_base64_to_wav_bytes(x)
        with io.BytesIO(wav_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=True)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return np.asarray(audio, dtype=np.float32), int(sr)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if source_sr != target_sr:
        waveform = librosa.resample(
            y=waveform,
            orig_sr=int(source_sr),
            target_sr=int(target_sr),
        )
    return np.asarray(waveform, dtype=np.float32)
