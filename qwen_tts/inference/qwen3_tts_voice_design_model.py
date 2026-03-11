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
    GenerateOptionsExtended,
    GenerateExtraArg,
    Qwen3TTSBaseModel,
    TTSInput,
    TTSInputs,
)


class Qwen3TTSVoiceDesignModel(Qwen3TTSBaseModel):
    model: Qwen3TTSVoiceDesignForConditionalGeneration

    def _generate_voice_design_part_wavs(
        self,
        tts_input: TTSInput,
        language: str,
        non_streaming_mode: bool,
        gen_kwargs: GenerateOptionsExtended,
    ) -> tuple[list[np.ndarray], int]:
        tokenized_parts = self._tokenize_tts_input(tts_input)
        context_ref_code: torch.Tensor | None = None
        context_ref_id: torch.Tensor | None = None
        part_wavs: list[np.ndarray] = []
        fs_out: int | None = None

        for tokenized_part in tokenized_parts:
            if context_ref_code is None or context_ref_id is None:
                talker_codes, _ = self.model.generate_voice_design(
                    input_role_id=tokenized_part.input_role_id,
                    input_text_id=tokenized_part.input_text_id,
                    instruct_id=tokenized_part.instruction_id,
                    language=language,
                    non_streaming_mode=non_streaming_mode,
                    **gen_kwargs,
                )
            else:
                talker_codes, _ = self.model.generate_voice_design(
                    input_role_id=tokenized_part.input_role_id,
                    input_text_id=tokenized_part.input_text_id,
                    instruct_id=tokenized_part.instruction_id,
                    language=language,
                    ref_code=context_ref_code,
                    ref_id=context_ref_id,
                    use_icl_prompt=True,
                    non_streaming_mode=non_streaming_mode,
                    **gen_kwargs,
                )

            wavs, fs = self._decode_talker_codes_batch([talker_codes])
            part_wavs.append(wavs[0])
            fs_out = fs if fs_out is None else fs_out
            if fs_out != fs:
                raise RuntimeError(
                    "Inconsistent sample rates returned during generation."
                )

            context_ref_code = self._concat_code_context(context_ref_code, talker_codes)
            context_ref_id = self._concat_text_context(
                context_ref_id,
                tokenized_part.input_text_id,
            )

        if fs_out is None:
            raise RuntimeError("No audio was generated from `tts_input`.")
        return part_wavs, fs_out

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
        normalized_tts_input = self._normalize_tts_input(tts_input)

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
        return self._generate_voice_design_part_wavs(
            tts_input=normalized_tts_input,
            language=language_value,
            non_streaming_mode=non_streaming_mode,
            gen_kwargs=gen_kwargs,
        )

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
        normalized_tts_inputs = self._normalize_tts_inputs(tts_inputs)
        languages = self._normalize_language_values(
            language, len(normalized_tts_inputs)
        )
        self._validate_languages(languages)

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
        wavs_out: list[list[np.ndarray]] = []
        fs_out: int | None = None
        for request_tts_input, request_language in zip(
            normalized_tts_inputs,
            languages,
            strict=True,
        ):
            part_wavs, fs = self._generate_voice_design_part_wavs(
                tts_input=request_tts_input,
                language=request_language,
                non_streaming_mode=non_streaming_mode,
                gen_kwargs=gen_kwargs,
            )
            wavs_out.append(part_wavs)
            fs_out = fs if fs_out is None else fs_out
            if fs_out != fs:
                raise RuntimeError(
                    "Inconsistent sample rates returned during generation."
                )

        if fs_out is None:
            raise RuntimeError("No audio was generated from `tts_inputs`.")
        return wavs_out, fs_out
