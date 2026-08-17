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
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import BatchFeature
from typing_extensions import override

from qwen_tts.audio_utils import load_audio_to_np_and_sr
from qwen_tts.core.models import Qwen3TTSConfig, mel_spectrogram

AudioLike = (
    str  # wav path, URL, base64
    | np.ndarray  # waveform (requires sr)
    | tuple[np.ndarray, int]  # (waveform, sr)
)

AudioCodes = list[list[int]]


class RawTTSJsonRow(TypedDict):
    audio: str
    text: str
    ref_audio: str


class PreparedTTSJsonRow(RawTTSJsonRow):
    audio_codes: AudioCodes


class TTSSample(TypedDict):
    text_ids: torch.Tensor
    audio_codes: torch.Tensor
    ref_mel: torch.Tensor


class TTSBatch(TypedDict):
    input_ids: torch.Tensor
    ref_mels: torch.Tensor
    attention_mask: torch.Tensor
    text_embedding_mask: torch.Tensor
    codec_embedding_mask: torch.Tensor
    codec_0_labels: torch.Tensor
    codec_ids: torch.Tensor
    codec_mask: torch.Tensor


class TTSProcessor(Protocol):
    def __call__(
        self,
        *,
        text: str,
        return_tensors: Literal["pt"],
        padding: Literal[True],
    ) -> BatchFeature: ...


def _json_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object.")

    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings.")
        parsed[key] = item
    return parsed


def _required_json_string(row: Mapping[str, object], key: str, context: str) -> str:
    if key not in row:
        raise KeyError(f"{context} is missing required key {key!r}.")
    value = row[key]
    if not isinstance(value, str):
        raise TypeError(f"{context}[{key!r}] must be a string.")
    return value


def parse_audio_codes(value: object, context: str) -> AudioCodes:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list of code frames.")

    audio_codes: AudioCodes = []
    for frame_index, frame_raw in enumerate(value):
        if not isinstance(frame_raw, list):
            raise TypeError(f"{context}[{frame_index}] must be a list.")
        frame: list[int] = []
        for code_index, code in enumerate(frame_raw):
            if not isinstance(code, int) or isinstance(code, bool):
                raise TypeError(
                    f"{context}[{frame_index}][{code_index}] must be an integer."
                )
            frame.append(code)
        audio_codes.append(frame)
    return audio_codes


def parse_raw_tts_json_row(value: object, context: str) -> RawTTSJsonRow:
    row = _json_object(value, context)
    row["audio"] = _required_json_string(row, "audio", context)
    row["text"] = _required_json_string(row, "text", context)
    row["ref_audio"] = _required_json_string(row, "ref_audio", context)
    return cast(RawTTSJsonRow, row)


def parse_prepared_tts_json_row(value: object, context: str) -> PreparedTTSJsonRow:
    row = _json_object(value, context)
    row["audio"] = _required_json_string(row, "audio", context)
    row["text"] = _required_json_string(row, "text", context)
    row["ref_audio"] = _required_json_string(row, "ref_audio", context)
    if "audio_codes" not in row:
        raise KeyError(f"{context} is missing required key 'audio_codes'.")
    row["audio_codes"] = parse_audio_codes(
        row["audio_codes"], f"{context}['audio_codes']"
    )
    return cast(PreparedTTSJsonRow, row)


def add_audio_codes(row: RawTTSJsonRow, audio_codes: AudioCodes) -> PreparedTTSJsonRow:
    prepared: dict[str, object] = dict(row)
    prepared["audio_codes"] = audio_codes
    return cast(PreparedTTSJsonRow, prepared)


class TTSDataset(Dataset[TTSSample]):
    _ASSISTANT_PREFIX = "<|im_start|>assistant\n"

    def __init__(
        self,
        data_list: Sequence[PreparedTTSJsonRow],
        processor: TTSProcessor,
        config: Qwen3TTSConfig,
        lag_num: int = -1,
    ) -> None:
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self._assistant_prefix_ids = self._tokenize_text(self._ASSISTANT_PREFIX)
        if self._assistant_prefix_ids.shape[1] != 3:
            raise ValueError(
                "The text tokenizer produced an unsupported Qwen TTS assistant prefix."
            )

    def __len__(self) -> int:
        return len(self.data_list)

    def _normalize_audio_inputs(
        self, audios: AudioLike | list[AudioLike]
    ) -> list[tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
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
        for i, (audio, sr) in enumerate(out):
            if audio.ndim > 1:
                out[i] = (np.mean(audio, axis=-1).astype(np.float32), sr)
        return out

    def _tokenize_text(self, text: str) -> torch.Tensor:
        input_data_raw: object = self.processor(
            text=text, return_tensors="pt", padding=True
        )
        if not isinstance(input_data_raw, BatchFeature):
            raise TypeError("Processor output must be a BatchFeature.")
        input_ids_raw: object = input_data_raw.get("input_ids")
        if not isinstance(input_ids_raw, torch.Tensor):
            raise TypeError("Processor output `input_ids` must be a tensor.")
        input_id = input_ids_raw
        return input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

    def _tokenize_assistant_input(self, text: str) -> torch.Tensor:
        return torch.cat((self._assistant_prefix_ids, self._tokenize_text(text)), dim=1)

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            raise ValueError("Only 24kHz reference audio is supported.")
        if audio.ndim != 1:
            raise ValueError("Reference audio must be a mono waveform.")
        if audio.size == 0:
            raise ValueError("Reference audio must not be empty.")
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    @override
    def __getitem__(self, index: int) -> TTSSample:
        item = self.data_list[index]

        text = item["text"]
        audio_codes = item["audio_codes"]
        ref_audio_path = item["ref_audio"]

        text_ids = self._tokenize_assistant_input(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        if audio_codes.dim() != 2 or audio_codes.shape[1] != 16:
            raise ValueError("`audio_codes` must have shape (frames, 16).")

        normalized = self._normalize_audio_inputs(ref_audio_path)
        wav, sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids,  # 1 , t
            "audio_codes": audio_codes,  # t, 16
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch: Sequence[TTSSample]) -> TTSBatch:
        if self.lag_num != -1:
            raise ValueError("Only lag_num=-1 is supported.")
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch.")

        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch
        ]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
        codec_token_ids = self.config.talker_config.codec_special_token_ids

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    codec_token_ids.no_think,
                    codec_token_ids.think_bos,
                    codec_token_ids.think_eos,
                    0,  # for speaker embedding
                    codec_token_ids.pad,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = codec_token_ids.pad
            input_ids[i, 8 + text_ids_len - 3, 1] = codec_token_ids.pad
            input_ids[i, 8 + text_ids_len - 2, 1] = codec_token_ids.bos
            input_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1
            ] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = codec_token_ids.eos

            codec_0_labels[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                codec_token_ids.eos
            )

            codec_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :
            ] = audio_codecs

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # for speaker embedding

            codec_mask[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        ref_mels = [data["ref_mel"] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
