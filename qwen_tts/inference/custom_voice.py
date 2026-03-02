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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .base import Qwen3TTSModelBase


class Qwen3TTSCustomVoice(Qwen3TTSModelBase):
    """Standalone wrapper for Qwen3 TTS CustomVoice model: generate_custom_voice."""

    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id, optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`,
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6":  # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

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

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs
