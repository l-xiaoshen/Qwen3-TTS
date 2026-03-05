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

from typing import Optional

import torch

from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_types import (
    GenerateConfigPrimitive,
    GenerationFeatureItem,
    VoiceClonePrompt,
    VoiceClonePromptSingle,
)


class Qwen3TTSVoiceCloneForConditionalGeneration(Qwen3TTSConditionalGenerationBase):
    @torch.no_grad()
    def generate_voice_clone(
        self,
        input_id: Optional[torch.Tensor] = None,
        instruct_id: Optional[torch.Tensor] = None,
        ref_id: Optional[torch.Tensor] = None,
        voice_clone_prompt: Optional[VoiceClonePromptSingle] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
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
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_id = self._validate_input_id(input_id)
        if voice_clone_prompt is None:
            raise ValueError(
                "`voice_clone_prompt` is required for voice clone generation."
            )

        instruct_id = self._normalize_instruct_id(instruct_id)
        ref_id = self._normalize_ref_id(ref_id)
        language = self._normalize_language(language)
        speaker = self._normalize_speaker(speaker)
        self._validate_voice_clone_prompt(voice_clone_prompt)
        _ = kwargs

        voice_clone_spk_embed = self.generate_speaker_prompt(
            voice_clone_prompt["ref_spk_embedding"]
        )
        feature_item = GenerationFeatureItem(
            speaker=speaker,
            speaker_embed=self._resolve_voice_clone_speaker_embed(
                voice_clone_prompt, voice_clone_spk_embed
            ),
            ref_code=voice_clone_prompt["ref_code"],
            ref_id=ref_id,
            use_icl_prompt=bool(voice_clone_prompt["icl_mode"]),
        )
        return self._generate_from_feature_item(
            input_id=input_id,
            instruct_id=instruct_id,
            language=language,
            feature_item=feature_item,
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
    def generate_voice_clone_batch(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor | None]] = None,
        ref_ids: Optional[list[torch.Tensor | None]] = None,
        voice_clone_prompt: Optional[VoiceClonePrompt] = None,
        languages: Optional[list[str]] = None,
        speakers: Optional[list[str | None]] = None,
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
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_ids = self._validate_input_ids_batch(input_ids)
        if voice_clone_prompt is None:
            raise ValueError(
                "`voice_clone_prompt` is required for voice clone generation."
            )
        batch_size = len(input_ids)
        instruct_ids = self._normalize_instruct_ids_batch(instruct_ids, batch_size)
        ref_ids = self._normalize_ref_ids_batch(ref_ids, batch_size)
        languages = self._normalize_languages_batch(languages, batch_size)
        speakers = self._normalize_speakers_batch(speakers, batch_size)
        self._validate_voice_clone_prompt_batch(voice_clone_prompt, batch_size)
        _ = kwargs

        voice_clone_spk_embeds = self.generate_speaker_prompt_batch(voice_clone_prompt)
        feature_items: list[GenerationFeatureItem] = []
        for index, (speaker, ref_id) in enumerate(zip(speakers, ref_ids)):
            feature_items.append(
                GenerationFeatureItem(
                    speaker=speaker,
                    speaker_embed=self._resolve_voice_clone_speaker_embed_batch(
                        index, voice_clone_prompt, voice_clone_spk_embeds
                    ),
                    ref_code=voice_clone_prompt["ref_code"][index],
                    ref_id=ref_id,
                    use_icl_prompt=bool(voice_clone_prompt["icl_mode"][index]),
                )
            )

        return self._generate_from_feature_items_batch(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            feature_items=feature_items,
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
    "Qwen3TTSVoiceCloneForConditionalGeneration",
]
