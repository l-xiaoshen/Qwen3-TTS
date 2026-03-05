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


class Qwen3TTSCustomVoiceModel(Qwen3TTSBaseModel):
    @torch.no_grad()
    def generate_custom_voice_batch(
        self,
        text: Union[str, list[str]],
        speaker: Union[str, list[str]],
        language: Optional[Union[str, list[str]]] = None,
        instruct: Optional[Union[str, list[str]]] = None,
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
        Generate speech with the CustomVoice model using a predefined speaker id.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice_batch")

        texts = self._ensure_list(text)
        if isinstance(language, list):
            languages: list[str] = []
            for item in language:
                if not isinstance(item, str):
                    raise TypeError("`language` list items must be strings.")
                languages.append(item)
        elif language is None:
            languages = ["Auto"] * len(texts)
        else:
            languages = [language] * len(texts)

        if isinstance(speaker, list):
            speakers: list[str | None] = []
            for item in speaker:
                if not isinstance(item, str):
                    raise TypeError("`speaker` list items must be strings.")
                speakers.append(item)
        else:
            speakers = [speaker]

        model_size = self.model.tts_model_size
        if isinstance(model_size, str) and model_size in "0b6":
            # for 0b6 model, instruct is not supported
            instructs: list[str | None] = [None] * len(texts)
        elif isinstance(instruct, list):
            instructs = []
            for item in instruct:
                if not isinstance(item, str):
                    raise TypeError("`instruct` list items must be strings.")
                instructs.append(item)
        elif instruct is None:
            instructs = [None] * len(texts)
        else:
            instructs = [instruct] * len(texts)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

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
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        speech_tokenizer = self._require_speech_tokenizer()
        wavs, fs = speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        return wavs, fs
