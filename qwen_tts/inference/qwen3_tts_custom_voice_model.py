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

from qwen_tts.core import SpeakerConfiguration

from ..core.models import Qwen3TTSCustomVoiceForConditionalGeneration
from .qwen3_tts_base_model import GenerateExtraArg, Qwen3TTSBaseModel


class Qwen3TTSCustomVoiceModel(Qwen3TTSBaseModel):
    model: Qwen3TTSCustomVoiceForConditionalGeneration

    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: str,
        speaker: SpeakerConfiguration,
        language: str = "Auto",
        instruct: str = "",
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        max_new_tokens: int = 2048,
        **kwargs: GenerateExtraArg,
    ) -> tuple[np.ndarray, int]:
        """
        Generate one utterance with the CustomVoice model.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice")

        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if not isinstance(instruct, str):
            raise TypeError("`instruct` must be a string.")

        language_value = self._normalize_language_value(language)
        speaker_value = speaker

        model_size = self.model.tts_model_size
        if isinstance(model_size, str) and model_size in "0b6":
            # for 0b6 model, instruct is not supported
            instruct_value = ""
        else:
            instruct_value = instruct

        self._validate_languages([language_value])
        self._validate_speaker_configuration(speaker_value)

        input_id = self._tokenize_assistant_input(text)
        instruct_id = self._tokenize_instruct(instruct_value)

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

        talker_codes, _ = self.model.generate_custom_voice(
            input_id=input_id,
            instruct_id=instruct_id,
            language=language_value,
            speaker=speaker_value,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self._decode_talker_codes_batch([talker_codes])
        return wavs[0], fs

    @torch.no_grad()
    def generate_custom_voice_batch(
        self,
        text: list[str],
        speaker: Sequence[SpeakerConfiguration],
        language: Sequence[str] = (),
        instruct: Sequence[str] = (),
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        max_new_tokens: int = 2048,
        **kwargs: GenerateExtraArg,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice_batch")

        if not isinstance(text, list):
            raise TypeError("`text` must be a list of strings.")
        texts: list[str] = []
        for item in text:
            if not isinstance(item, str):
                raise TypeError("`text` list items must be strings.")
            texts.append(item)

        languages = self._normalize_language_values(language, len(texts))

        if not isinstance(speaker, Sequence) or isinstance(speaker, (str, bytes)):
            raise TypeError(
                "`speaker` must be a sequence of speaker configuration dictionaries."
            )
        speakers = list(speaker)

        model_size = self.model.tts_model_size
        if isinstance(model_size, str) and model_size in "0b6":
            # for 0b6 model, instruct is not supported
            instructs = [""] * len(texts)
        elif not isinstance(instruct, Sequence) or isinstance(instruct, (str, bytes)):
            raise TypeError("`instruct` must be a sequence of strings.")
        elif len(instruct) == 0:
            instructs = [""] * len(texts)
        else:
            instructs = []
            for item in instruct:
                if not isinstance(item, str):
                    raise TypeError("`instruct` list items must be strings.")
                instructs.append(item)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speaker_configurations(speakers, len(texts))

        input_ids = self._tokenize_assistant_inputs(texts)
        instruct_ids = self._tokenize_instructs(instructs)

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

        talker_codes_list, _ = self.model.generate_custom_voice_batch(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        return self._decode_talker_codes_batch(talker_codes_list)
