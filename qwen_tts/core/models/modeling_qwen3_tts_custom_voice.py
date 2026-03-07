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
"""Custom-voice conditional-generation model for Qwen3 TTS."""

from collections.abc import Sequence

import torch

from qwen_tts.core import SpeakerConfiguration

from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_types import (
    GenerateConfigPrimitive,
    SubTalkerConfiguration,
)


class Qwen3TTSCustomVoiceForConditionalGeneration(Qwen3TTSConditionalGenerationBase):
    @torch.no_grad()
    def generate_custom_voice(
        self,
        input_id: torch.Tensor,
        instruct_id: torch.Tensor | None,
        language: str,
        speaker: SpeakerConfiguration | torch.Tensor,
        ref_code: torch.Tensor | None = None,
        ref_id: torch.Tensor | None = None,
        use_icl_prompt: bool = False,
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
        if len(kwargs) != 0:
            raise TypeError(f"Unsupported generation kwargs: {sorted(kwargs)}")

        if isinstance(speaker, torch.Tensor):
            language_id = self._resolve_language_id(language, "")
            speaker_embed = speaker.to(
                device=self.talker.device, dtype=self.talker.dtype
            )
        else:
            language_id = self._resolve_language_id_for_speaker_config(
                language, speaker
            )
            speaker_embed = self._resolve_custom_voice_speaker_embed(
                speaker, input_id.dtype
            )
        return self._generate_voice_clone_from_ids(
            input_id=input_id,
            instruct_id=instruct_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
            non_streaming_mode=non_streaming_mode,
            ref_code=ref_code,
            ref_id=ref_id,
            use_icl_prompt=use_icl_prompt,
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
    def generate_custom_voice_batch(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: Sequence[torch.Tensor | None] = (),
        languages: Sequence[str] = (),
        speakers: Sequence[SpeakerConfiguration | torch.Tensor] = (),
        ref_codes: Sequence[torch.Tensor | None] = (),
        ref_ids: Sequence[torch.Tensor | None] = (),
        use_icl_prompts: Sequence[bool] = (),
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
        languages = self._normalize_languages_batch(languages, batch_size)
        ref_ids = self._normalize_ref_ids_batch(ref_ids, batch_size)
        if len(speakers) == 0:
            speakers = [{} for _ in range(batch_size)]
        elif len(speakers) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, speakers={len(speakers)}"
            )
        else:
            speakers = list(speakers)
        if len(ref_codes) == 0:
            ref_codes = [None] * batch_size
        elif len(ref_codes) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, ref_codes={len(ref_codes)}"
            )
        else:
            ref_codes = list(ref_codes)
        if len(use_icl_prompts) == 0:
            use_icl_prompts = [False] * batch_size
        elif len(use_icl_prompts) != batch_size:
            raise ValueError(
                "Batch size mismatch: "
                f"input_ids={batch_size}, use_icl_prompts={len(use_icl_prompts)}"
            )
        else:
            use_icl_prompts = [bool(flag) for flag in use_icl_prompts]
        if len(kwargs) != 0:
            raise TypeError(f"Unsupported generation kwargs: {sorted(kwargs)}")

        language_ids: list[int | None] = []
        speaker_embeds: list[torch.Tensor | None] = []
        for input_id, language, speaker in zip(input_ids, languages, speakers):
            if isinstance(speaker, torch.Tensor):
                language_ids.append(self._resolve_language_id(language, ""))
                speaker_embeds.append(
                    speaker.to(device=self.talker.device, dtype=self.talker.dtype)
                )
            else:
                language_ids.append(
                    self._resolve_language_id_for_speaker_config(language, speaker)
                )
                speaker_embeds.append(
                    self._resolve_custom_voice_speaker_embed(speaker, input_id.dtype)
                )

        return self._generate_voice_clone_batch_from_ids(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language_ids=language_ids,
            speaker_embeds=speaker_embeds,
            ref_codes=list(ref_codes),
            ref_ids=ref_ids,
            use_icl_prompts=list(use_icl_prompts),
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
    "Qwen3TTSCustomVoiceForConditionalGeneration",
]
