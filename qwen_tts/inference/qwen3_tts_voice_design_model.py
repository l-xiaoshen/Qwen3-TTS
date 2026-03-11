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
from .qwen3_tts_base_model import (
    GenerateExtraArg,
    Qwen3TTSBaseModel,
    TTSInput,
    TTSInputs,
)


class Qwen3TTSVoiceDesignModel(Qwen3TTSBaseModel):
    model: Qwen3TTSVoiceDesignForConditionalGeneration

    @torch.no_grad()
    def generate_voice_design(
        self,
        tts_input: TTSInput,
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
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate one shared-context multi-part utterance with the VoiceDesign model.
        """
        self._ensure_model_type("voice_design", "generate_voice_design")
        plan = self._build_tts_input_plan(tts_input)
        split_outputs = self._should_split_part_outputs(plan, non_streaming_mode)

        language_value = self._normalize_language_value(language)
        self._validate_languages([language_value])

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
            input_role_id=plan.input_role_id,
            input_text_id=plan.input_text_id,
            instruct_id=plan.instruct_id,
            language=language_value,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )
        wavs, fs = self._decode_talker_codes_batch([talker_codes])
        if not split_outputs:
            return [wavs[0]], fs
        return self._split_wav_by_plan(wavs[0], talker_codes, plan), fs

    @torch.no_grad()
    def generate_voice_design_batch(
        self,
        tts_inputs: TTSInputs,
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
    ) -> tuple[list[list[np.ndarray]], int]:
        """
        Generate shared-context multi-part speech for a batch of requests.
        """
        self._ensure_model_type("voice_design", "generate_voice_design_batch")
        plans = self._build_tts_input_plans(tts_inputs)
        languages = self._normalize_language_values(language, len(plans))
        self._validate_languages(languages)
        split_outputs = [
            self._should_split_part_outputs(plan, non_streaming_mode) for plan in plans
        ]

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
            input_role_ids=[plan.input_role_id for plan in plans],
            input_text_ids=[plan.input_text_id for plan in plans],
            instruct_ids=[plan.instruct_id for plan in plans],
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )
        wavs, fs = self._decode_talker_codes_batch(talker_codes_list)
        wavs_out = [
            (
                self._split_wav_by_plan(wav, talker_codes, plan)
                if split_output
                else [wav]
            )
            for wav, talker_codes, plan, split_output in zip(
                wavs, talker_codes_list, plans, split_outputs, strict=True
            )
        ]
        return wavs_out, fs
