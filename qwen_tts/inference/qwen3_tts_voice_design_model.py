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
from collections.abc import Sequence

import numpy as np
import torch

from ..core.models import (
    Qwen3TTSVoiceDesignForConditionalGeneration,
    SubTalkerConfiguration,
)
from .qwen3_tts_base_model import GenerateExtraArg, Qwen3TTSBaseModel


class Qwen3TTSVoiceDesignModel(Qwen3TTSBaseModel):
    model: Qwen3TTSVoiceDesignForConditionalGeneration

    @torch.no_grad()
    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "Auto",
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int = 2048,
        **kwargs: GenerateExtraArg,
    ) -> tuple[np.ndarray, int]:
        """
        Generate one utterance with the VoiceDesign model.
        """
        self._ensure_model_type("voice_design", "generate_voice_design")

        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if not isinstance(instruct, str):
            raise TypeError("`instruct` must be a string.")

        language_value = self._normalize_language_value(language)
        self._validate_languages([language_value])

        segments = self._build_runtime_segments(text=text, instruct=instruct)
        input_role_id, input_text_id = self._tokenize_assistant_segments(segments)
        instruct_id = self._tokenize_instruction_segments(segments)

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes, _ = self.model.generate_voice_design(
            input_role_id=input_role_id,
            input_text_id=input_text_id,
            instruct_id=instruct_id,
            language=language_value,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self._decode_talker_codes_batch([talker_codes])
        return wavs[0], fs

    @torch.no_grad()
    def generate_voice_design_batch(
        self,
        text: list[str],
        instruct: list[str],
        language: Sequence[str] = (),
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int = 2048,
        **kwargs: GenerateExtraArg,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.
        """
        self._ensure_model_type("voice_design", "generate_voice_design_batch")

        if not isinstance(text, list):
            raise TypeError("`text` must be a list of strings.")
        texts: list[str] = []
        for item in text:
            if not isinstance(item, str):
                raise TypeError("`text` list items must be strings.")
            texts.append(item)

        if not isinstance(instruct, list):
            raise TypeError("`instruct` must be a list of strings.")
        instructs: list[str] = []
        for item in instruct:
            if not isinstance(item, str):
                raise TypeError("`instruct` list items must be strings.")
            instructs.append(item)

        languages = self._normalize_language_values(language, len(texts))

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)

        input_role_ids: list[torch.Tensor] = []
        input_text_ids: list[torch.Tensor] = []
        instruct_ids: list[torch.Tensor | None] = []
        for text_value, instruct_value in zip(texts, instructs):
            segments = self._build_runtime_segments(
                text=text_value,
                instruct=instruct_value,
            )
            input_role_id, input_text_id = self._tokenize_assistant_segments(segments)
            input_role_ids.append(input_role_id)
            input_text_ids.append(input_text_id)
            instruct_ids.append(self._tokenize_instruction_segments(segments))

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes_list, _ = self.model.generate_voice_design_batch(
            input_role_ids=input_role_ids,
            input_text_ids=input_text_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        return self._decode_talker_codes_batch(talker_codes_list)
