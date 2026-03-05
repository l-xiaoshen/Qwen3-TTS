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
"""Validation and per-feature normalization helpers."""

from typing import Optional

import torch

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
from .modeling_qwen3_tts_types import VoiceClonePrompt


class Qwen3TTSGenerationValidationMixin:
    config: Qwen3TTSConfig
    talker: Qwen3TTSTalkerForConditionalGeneration

    def _validate_input_ids_batch(
        self, input_ids: Optional[list[torch.Tensor]]
    ) -> list[torch.Tensor]:
        if input_ids is None or len(input_ids) == 0:
            raise ValueError("`input_ids` must be a non-empty list of tensors.")
        return input_ids

    def _normalize_languages_batch(
        self, languages: Optional[list[str]], batch_size: int
    ) -> list[str]:
        if languages is None:
            return ["auto"] * batch_size
        if len(languages) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, languages={len(languages)}"
            )
        return languages

    def _normalize_speakers_batch(
        self, speakers: Optional[list[str | None]], batch_size: int
    ) -> list[str | None]:
        if speakers is None:
            return [None] * batch_size
        if len(speakers) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, speakers={len(speakers)}"
            )
        return speakers

    def _normalize_instruct_ids_batch(
        self,
        instruct_ids: Optional[list[torch.Tensor | None]],
        batch_size: int,
    ) -> list[torch.Tensor | None]:
        if instruct_ids is None:
            return [None] * batch_size
        if len(instruct_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, instruct_ids={len(instruct_ids)}"
            )
        return instruct_ids

    def _normalize_ref_ids_batch(
        self, ref_ids: Optional[list[torch.Tensor | None]], batch_size: int
    ) -> list[torch.Tensor | None]:
        if ref_ids is None:
            return [None] * batch_size
        if len(ref_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, ref_ids={len(ref_ids)}"
            )
        return ref_ids

    def _validate_voice_clone_prompt_batch(
        self, voice_clone_prompt: VoiceClonePrompt, batch_size: int
    ) -> None:
        if not (
            len(voice_clone_prompt["ref_code"])
            == len(voice_clone_prompt["ref_spk_embedding"])
            == len(voice_clone_prompt["x_vector_only_mode"])
            == len(voice_clone_prompt["icl_mode"])
            == batch_size
        ):
            raise ValueError(
                "Batch size mismatch in `voice_clone_prompt` fields and `input_ids`."
            )

    def _resolve_custom_voice_speaker_embed(
        self, speaker: str | None, input_dtype: torch.dtype
    ) -> torch.Tensor | None:
        if speaker == "" or speaker is None:
            return None
        speaker_lower = speaker.lower()
        spk_id_map = self.config.talker_config.spk_id
        if spk_id_map is None or speaker_lower not in spk_id_map:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        spk_id = spk_id_map[speaker_lower]
        return self.talker.get_input_embeddings()(
            torch.tensor(
                spk_id,
                device=self.talker.device,
                dtype=input_dtype,
            )
        )

    def _resolve_voice_clone_speaker_embed(
        self,
        index: int,
        voice_clone_prompt: VoiceClonePrompt,
        voice_clone_spk_embeds: list[torch.Tensor],
    ) -> torch.Tensor | None:
        if (
            voice_clone_prompt["x_vector_only_mode"][index]
            or voice_clone_prompt["icl_mode"][index]
        ):
            return voice_clone_spk_embeds[index]
        return None

    def _resolve_language_id(self, language: str, speaker: str | None) -> int | None:
        language_lower = language.lower()
        language_map = self.config.talker_config.codec_language_id
        if language_map is None:
            raise NotImplementedError("Language map is not available in config")
        if language_lower == "auto":
            language_id = None
        else:
            if language_lower not in language_map:
                raise NotImplementedError(f"Language {language} not implemented")
            language_id = language_map[language_lower]

        dialect_map = self.config.talker_config.spk_is_dialect or {}
        dialect_value = (
            dialect_map.get(speaker.lower(), False)
            if speaker is not None and speaker != ""
            else False
        )
        if (
            language_lower in ["chinese", "auto"]
            and speaker != ""
            and speaker is not None
            and dialect_value is not False
            and isinstance(dialect_value, str)
            and dialect_value in language_map
        ):
            language_id = language_map[dialect_value]
        return language_id


__all__ = [
    "Qwen3TTSGenerationValidationMixin",
]
