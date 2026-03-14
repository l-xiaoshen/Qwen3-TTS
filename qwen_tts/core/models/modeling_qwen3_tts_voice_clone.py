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
"""Voice-clone conditional-generation model for Qwen3 TTS."""

from collections.abc import Sequence

import torch

from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_types import (
    GenerateConfigPrimitive,
    SubTalkerConfiguration,
    VoiceClonePrompt,
    VoiceClonePromptSingle,
)


class Qwen3TTSVoiceCloneForConditionalGeneration(Qwen3TTSConditionalGenerationBase):
    @torch.no_grad()
    def generate_voice_clone(
        self,
        input_id: torch.Tensor,
        voice_clone_prompt: VoiceClonePromptSingle,
        instruct_id: torch.Tensor | None = None,
        ref_id: torch.Tensor | None = None,
        language: str = "Auto",
        speaker: str = "",
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_id = self._validate_input_id(input_id)
        instruct_id = self._normalize_instruct_id(instruct_id)
        ref_id = self._normalize_ref_id(ref_id)
        language = self._normalize_language(language)
        speaker = self._normalize_speaker(speaker)
        self._validate_voice_clone_prompt(voice_clone_prompt)
        if len(kwargs) != 0:
            raise TypeError(f"Unsupported generation kwargs: {sorted(kwargs)}")

        language_id = self._resolve_language_id(language, speaker)
        voice_clone_spk_embed = self.generate_speaker_prompt(
            voice_clone_prompt["ref_spk_embedding"]
        )
        speaker_embed = self._resolve_voice_clone_speaker_embed(
            voice_clone_prompt, voice_clone_spk_embed
        )
        return self._generate_voice_clone_from_ids(
            input_id=input_id,
            instruct_id=instruct_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
            non_streaming_mode=non_streaming_mode,
            ref_code=voice_clone_prompt["ref_code"],
            ref_id=ref_id,
            use_icl_prompt=bool(voice_clone_prompt["icl_mode"]),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_configuration=subtalker_configuration,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

    @torch.no_grad()
    def generate_voice_clone_batch(
        self,
        input_ids: list[torch.Tensor],
        voice_clone_prompt: VoiceClonePrompt,
        instruct_ids: Sequence[torch.Tensor | None] = (),
        ref_ids: Sequence[torch.Tensor | None] = (),
        languages: Sequence[str] = (),
        speakers: Sequence[str] = (),
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_ids = self._validate_input_ids_batch(input_ids)
        batch_size = len(input_ids)
        instruct_ids = self._normalize_instruct_ids_batch(instruct_ids, batch_size)
        ref_ids = self._normalize_ref_ids_batch(ref_ids, batch_size)
        languages = self._normalize_languages_batch(languages, batch_size)
        speakers = self._normalize_speakers_batch(speakers, batch_size)
        self._validate_voice_clone_prompt_batch(voice_clone_prompt, batch_size)
        if len(kwargs) != 0:
            raise TypeError(f"Unsupported generation kwargs: {sorted(kwargs)}")

        language_ids = [
            self._resolve_language_id(language, speaker)
            for language, speaker in zip(languages, speakers)
        ]
        voice_clone_spk_embeds = self.generate_speaker_prompt_batch(voice_clone_prompt)
        speaker_embeds: list[torch.Tensor | None] = []
        for index in range(batch_size):
            speaker_embeds.append(
                self._resolve_voice_clone_speaker_embed_batch(
                    index, voice_clone_prompt, voice_clone_spk_embeds
                )
            )

        return self._generate_voice_clone_batch_from_ids(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language_ids=language_ids,
            speaker_embeds=speaker_embeds,
            ref_codes=voice_clone_prompt["ref_code"],
            ref_ids=ref_ids,
            use_icl_prompts=[
                bool(icl_mode) for icl_mode in voice_clone_prompt["icl_mode"]
            ],
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_configuration=subtalker_configuration,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )


__all__ = [
    "Qwen3TTSVoiceCloneForConditionalGeneration",
]
