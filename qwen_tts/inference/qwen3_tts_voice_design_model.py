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
from typing import Optional, Union

import numpy as np
import torch

from .qwen3_tts_base_model import GenerateExtraArg, Qwen3TTSBaseModel


class Qwen3TTSVoiceDesignModel(Qwen3TTSBaseModel):
    @torch.no_grad()
    def generate_voice_design_batch(
        self,
        text: Union[str, list[str]],
        instruct: Union[str, list[str]],
        language: Optional[Union[str, list[str]]] = None,
        non_streaming_mode: bool = True,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: GenerateExtraArg,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.
        """
        self._ensure_model_type("voice_design", "generate_voice_design_batch")

        texts = self._ensure_list(text)
        if isinstance(language, list):
            languages = list(language)
        elif language is None:
            languages = ["Auto"] * len(texts)
        else:
            languages = [language] * len(texts)
        instructs = list(instruct) if isinstance(instruct, list) else [instruct]

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)

        input_ids = self._tokenize_texts_batch([self._build_assistant_text(t) for t in texts])

        instruct_ids: list[torch.Tensor | None] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(
                    self._tokenize_texts_batch([self._build_instruct_text(ins)])[0]
                )

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes_list, _ = self.model.generate_batch(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        speech_tokenizer = self._require_speech_tokenizer()
        wavs, fs = speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        return wavs, fs
