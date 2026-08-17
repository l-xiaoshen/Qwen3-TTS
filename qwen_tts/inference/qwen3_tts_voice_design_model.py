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
import numpy as np
import torch

from ..core.models import (
    Qwen3TTSProcessor,
    Qwen3TTSVoiceDesignForConditionalGeneration,
    SubTalkerConfiguration,
)
from .qwen3_tts_base_model import (
    GenerationDefaults,
    Qwen3TTSBaseModel,
    TTSBatchInput,
    TTSInput,
)

StringBatchInput = list[str] | tuple[str, ...]


class Qwen3TTSVoiceDesignModel(Qwen3TTSBaseModel):
    model: Qwen3TTSVoiceDesignForConditionalGeneration
    _model_class = Qwen3TTSVoiceDesignForConditionalGeneration

    def __init__(
        self,
        model: Qwen3TTSVoiceDesignForConditionalGeneration,
        processor: Qwen3TTSProcessor,
        generate_defaults: GenerationDefaults | None = None,
    ) -> None:
        super().__init__(model, processor, generate_defaults)

    @torch.no_grad()
    def generate_voice_design(
        self,
        tts_input: TTSInput,
        *,
        language: str = "Auto",
        non_streaming_mode: bool = True,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate one assistant waveform per turn in a shared VoiceDesign context.
        """
        self._ensure_model_type("voice_design", "generate_voice_design")

        chunks = self._normalize_tts_input(tts_input)
        language_value = self._normalize_language_value(language)
        self._validate_languages([language_value])

        input_ids, instruct_ids = self._tokenize_tts_chunks(chunks)

        generation_options = self._resolve_generation_options(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        talker_codes_list, _ = self.model.generate_voice_design_turns(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language=language_value,
            non_streaming_mode=non_streaming_mode,
            **generation_options,
        )

        return self._decode_talker_turns(talker_codes_list)

    @torch.no_grad()
    def generate_voice_design_batch(
        self,
        tts_input: TTSBatchInput,
        *,
        language: StringBatchInput = (),
        non_streaming_mode: bool = True,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> tuple[list[list[np.ndarray]], int]:
        """
        Generate batched shared-context turns.
        """
        self._ensure_model_type("voice_design", "generate_voice_design_batch")

        structured_inputs = self._normalize_tts_batch_input(tts_input)
        languages = self._normalize_language_values(language, len(structured_inputs))

        if len(languages) != len(structured_inputs):
            raise ValueError(
                "Batch size mismatch: "
                f"tts_input={len(structured_inputs)}, language={len(languages)}"
            )

        self._validate_languages(languages)

        generation_options = self._resolve_generation_options(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        wavs_by_input: list[list[np.ndarray]] = []
        sample_rate: int | None = None
        for chunks, language_value in zip(structured_inputs, languages):
            input_ids, instruct_ids = self._tokenize_tts_chunks(chunks)
            talker_codes_list, _ = self.model.generate_voice_design_turns(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                language=language_value,
                non_streaming_mode=non_streaming_mode,
                **generation_options,
            )
            wavs, fs = self._decode_talker_turns(talker_codes_list)
            if sample_rate is not None and fs != sample_rate:
                raise RuntimeError("Decoded sample rates differ across batch items.")
            sample_rate = fs
            wavs_by_input.append(wavs)
        if sample_rate is None:
            raise RuntimeError("Structured batch generation produced no outputs.")
        return wavs_by_input, sample_rate
