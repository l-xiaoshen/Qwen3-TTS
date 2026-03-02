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
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from ..prompt_layout import (
    ASSISTANT_OPEN_SUFFIX_TOKEN_COUNT,
    ASSISTANT_PREFIX_TOKEN_COUNT,
    ASSISTANT_TRAILING_TEXT_START_TOKEN_INDEX,
)

if TYPE_CHECKING:
    from .voice_design import Qwen3TTSVoiceDesign


class VoiceDesignSingleStreamer:
    """
    Stateful VoiceDesign streamer for single-sample real-time generation.

    A streamer is created once with fixed language/instruct and keeps decoder
    state across repeated `stream()` calls.
    """

    def __init__(
        self,
        tts_model: "Qwen3TTSVoiceDesign",
        language: str,
        language_id: Optional[int],
        instruct: str,
        instruct_ids: Optional[torch.Tensor],
        non_streaming_mode: bool,
        generate_kwargs: Dict[str, Any],
    ) -> None:
        self._tts_model = tts_model
        self.language = language
        self.instruct = instruct
        self._language_id = language_id
        self._instruct_ids = instruct_ids
        self._non_streaming_mode = non_streaming_mode
        self._closed = False
        self._text_buffer = ""
        self._stream_state = None
        self._accumulated_input_ids: torch.Tensor = tts_model._tokenize_texts(
            ["<|im_start|>assistant\n"]
        )[0]
        self._all_talker_codes: torch.Tensor = torch.empty(
            (0, tts_model.model.config.talker_config.num_code_groups),
            dtype=torch.long,
        )
        self._emitted_samples: int = 0
        self.sample_rate: int = tts_model.model.speech_tokenizer.get_output_sample_rate()

        gen = dict(generate_kwargs)
        gen.pop("max_new_tokens", None)
        self._generate_kwargs = gen

    def _tokenize_and_append(self, text: str) -> None:
        """Tokenize *text* and append the resulting IDs to the accumulated input."""
        chunk_ids = self._tts_model._tokenize_texts([text])[0]
        if chunk_ids.shape[1] == 0:
            return
        self._accumulated_input_ids = torch.cat(
            [self._accumulated_input_ids, chunk_ids], dim=1,
        )

    def _remaining_text_steps(self, input_ids: torch.Tensor, include_eos: bool = False) -> int:
        """Return the number of text conditioning steps not yet generated.

        No suffix: trailing_text_hidden is built from
        input_id[:, ASSISTANT_TRAILING_TEXT_START_TOKEN_INDEX:], so text
        steps are computed relative to the shared prompt-layout constants.
        """
        offset = (
            ASSISTANT_PREFIX_TOKEN_COUNT
            if include_eos
            else ASSISTANT_TRAILING_TEXT_START_TOKEN_INDEX
        )
        text_steps = max(int(input_ids.shape[1]) - offset, 0)
        current = 0 if self._stream_state is None else max(int(self._stream_state.generation_step), 0)
        return max(text_steps - current, 0)

    def _run_generation_step(
        self,
        input_ids: torch.Tensor,
        suppress_eos: bool,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if self._stream_state is None:
            talker_codes, self._stream_state = self._tts_model.model.init_single_voice_design_stream(
                input_ids=input_ids,
                instruct_ids=self._instruct_ids,
                language_id=self._language_id,
                non_streaming_mode=self._non_streaming_mode,
                max_new_tokens=max_new_tokens,
                suppress_eos=suppress_eos,
                suffix_len=ASSISTANT_OPEN_SUFFIX_TOKEN_COUNT,
                **self._generate_kwargs,
            )
            return talker_codes

        if self._stream_state.finished:
            return torch.empty((0, self._tts_model.model.config.talker_config.num_code_groups), dtype=torch.long)

        talker_codes, self._stream_state = self._tts_model.model.continue_single_voice_design_stream(
            input_ids=input_ids,
            stream_state=self._stream_state,
            non_streaming_mode=self._non_streaming_mode,
            max_new_tokens=max_new_tokens,
            suppress_eos=suppress_eos,
            suffix_len=ASSISTANT_OPEN_SUFFIX_TOKEN_COUNT,
            **self._generate_kwargs,
        )
        return talker_codes

    def _decode_new_audio(self, talker_codes: torch.Tensor) -> np.ndarray:
        if talker_codes.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        self._all_talker_codes = torch.cat(
            [self._all_talker_codes.to(talker_codes.device), talker_codes],
            dim=0,
        )

        wavs, _ = self._tts_model.model.speech_tokenizer.decode([{"audio_codes": self._all_talker_codes}])
        full_wav = wavs[0]
        if self._emitted_samples >= len(full_wav):
            return np.zeros((0,), dtype=np.float32)
        out = full_wav[self._emitted_samples:]
        self._emitted_samples = len(full_wav)
        return out

    @torch.no_grad()
    def stream(self, text_chunk: str) -> np.ndarray:
        """
        Consume one text chunk (any length) and return newly generated wav samples.
        """
        if self._closed:
            raise RuntimeError("Streamer is closed. Create a new streamer to continue.")
        if not isinstance(text_chunk, str):
            raise TypeError(f"text_chunk must be str, got {type(text_chunk)}")
        if text_chunk == "":
            return np.zeros((0,), dtype=np.float32)
        self._text_buffer += text_chunk

        self._tokenize_and_append(text_chunk)
        input_ids = self._accumulated_input_ids
        remaining = self._remaining_text_steps(input_ids, include_eos=False)
        if remaining <= 0:
            return np.zeros((0,), dtype=np.float32)
        talker_codes = self._run_generation_step(
            input_ids=input_ids,
            suppress_eos=True,
            max_new_tokens=remaining,
        )
        return self._decode_new_audio(talker_codes)

    @torch.no_grad()
    def finalize(self) -> np.ndarray:
        """
        Generate remaining audio with EOS enabled and return newly generated wav.

        Call repeatedly until an empty array is returned.
        """
        if self._closed:
            raise RuntimeError("Streamer is closed. Create a new streamer to continue.")
        if self._text_buffer == "":
            return np.zeros((0,), dtype=np.float32)
        if self._stream_state is not None and self._stream_state.finished:
            return np.zeros((0,), dtype=np.float32)

        input_ids = self._accumulated_input_ids
        remaining = self._remaining_text_steps(input_ids, include_eos=True)
        talker_codes = self._run_generation_step(
            input_ids=input_ids,
            suppress_eos=False,
            max_new_tokens=max(remaining, 512),
        )
        return self._decode_new_audio(talker_codes)

    def flush(self) -> np.ndarray:
        """Alias of finalize() for streaming API naming compatibility."""
        return self.finalize()

    def close(self) -> None:
        """Release all streamer-held state."""
        if self._closed:
            return
        self._stream_state = None
        self._instruct_ids = None
        self._text_buffer = ""
        self._accumulated_input_ids = None
        self._all_talker_codes = None
        self._emitted_samples = 0
        self._tts_model.model.talker.rope_deltas = None
        self._closed = True
