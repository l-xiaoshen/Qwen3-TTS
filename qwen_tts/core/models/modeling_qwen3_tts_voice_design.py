# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Voice-design conditional-generation model for Qwen3 TTS."""

from collections.abc import Sequence
from typing import cast

import torch

from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_types import (
    GenerateConfigPrimitive,
)


class Qwen3TTSVoiceDesignForConditionalGeneration(Qwen3TTSConditionalGenerationBase):
    @torch.no_grad()
    def generate_voice_design(
        self,
        input_id: torch.Tensor,
        instruct_id: torch.Tensor | None = None,
        language: str = "auto",
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_id = self._validate_input_id(input_id)
        instruct_id = self._normalize_instruct_id(instruct_id)
        language = self._normalize_language(language)
        _ = kwargs

        language_id = self._resolve_language_id(language, "")
        return self._generate_standard_from_ids(
            input_id=input_id,
            instruct_id=instruct_id,
            language_id=language_id,
            speaker_embed=None,
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

    @torch.no_grad()
    def generate_voice_design_batch(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: Sequence[torch.Tensor | None] = (),
        languages: Sequence[str] = (),
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_ids = self._validate_input_ids_batch(input_ids)
        batch_size = len(input_ids)
        instruct_ids = self._normalize_instruct_ids_batch(instruct_ids, batch_size)
        languages = self._normalize_languages_batch(languages, batch_size)
        _ = kwargs

        language_ids = [
            self._resolve_language_id(language, "") for language in languages
        ]
        return self._generate_standard_batch_from_ids(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language_ids=language_ids,
            speaker_embeds=[
                cast(torch.Tensor | None, None) for _ in range(len(input_ids))
            ],
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )


__all__ = [
    "Qwen3TTSVoiceDesignForConditionalGeneration",
]
