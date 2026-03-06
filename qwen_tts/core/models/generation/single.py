# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
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
"""Single-sample generation helpers without batch collation overhead."""

from collections.abc import Sequence

import torch

from ..configuration_qwen3_tts import Qwen3TTSConfig
from ..modeling_qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
from ..modeling_qwen3_tts_types import VoiceClonePromptSingle
from .core import Qwen3TTSGenerationCoreMixin


class Qwen3TTSGenerationSingleMixin(Qwen3TTSGenerationCoreMixin):
    config: Qwen3TTSConfig
    talker: Qwen3TTSTalkerForConditionalGeneration

    def _validate_input_id(self, input_id: torch.Tensor) -> torch.Tensor:
        if input_id.dim() == 1:
            return input_id.unsqueeze(0)
        if input_id.dim() != 2:
            raise ValueError("`input_id` must be a 1D or 2D tensor.")
        return input_id

    def _normalize_language(self, language: str) -> str:
        return language

    def _normalize_speaker(self, speaker: str) -> str:
        return speaker

    def _normalize_instruct_id(
        self, instruct_id: torch.Tensor | None
    ) -> torch.Tensor | None:
        if instruct_id is not None and instruct_id.dim() == 1:
            return instruct_id.unsqueeze(0)
        return instruct_id

    def _normalize_ref_id(self, ref_id: torch.Tensor | None) -> torch.Tensor | None:
        if ref_id is not None and ref_id.dim() == 1:
            return ref_id.unsqueeze(0)
        return ref_id

    def _validate_voice_clone_prompt(
        self, voice_clone_prompt: VoiceClonePromptSingle
    ) -> None:
        required_keys = (
            "ref_code",
            "ref_spk_embedding",
            "x_vector_only_mode",
            "icl_mode",
        )
        for key in required_keys:
            if key not in voice_clone_prompt:
                raise KeyError(f"Missing key `{key}` in `voice_clone_prompt`.")

    def _resolve_voice_clone_speaker_embed(
        self,
        voice_clone_prompt: VoiceClonePromptSingle,
        voice_clone_spk_embed: torch.Tensor,
    ) -> torch.Tensor | None:
        if voice_clone_prompt["x_vector_only_mode"] or voice_clone_prompt["icl_mode"]:
            return voice_clone_spk_embed
        return None

    def _run_talker_generation(
        self,
        talker_input_embeds: list[torch.Tensor],
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        suppress_tokens: list[int],
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(talker_input_embeds) == 0:
            raise RuntimeError(
                "Generation input must contain at least one embed block."
            )
        talker_input_embed = torch.cat(talker_input_embeds, dim=1)
        talker_attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]),
            device=talker_input_embed.device,
            dtype=torch.long,
        )

        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            eos_token_id=(
                eos_token_id
                if eos_token_id is not None
                else self.config.talker_config.codec_eos_token_id
            ),
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

        hidden_states = getattr(talker_result, "hidden_states", None)
        if not isinstance(hidden_states, Sequence):
            raise RuntimeError(
                "Talker generate output does not contain `hidden_states`."
            )

        talker_code_steps: list[torch.Tensor] = []
        talker_hidden_steps: list[torch.Tensor] = []
        for step in hidden_states:
            if not isinstance(step, tuple) or len(step) == 0:
                continue
            codec_ids = step[-1]
            if isinstance(codec_ids, torch.Tensor):
                talker_code_steps.append(codec_ids)

            text_hidden_states = step[0]
            if (
                isinstance(text_hidden_states, tuple)
                and len(text_hidden_states) > 0
                and isinstance(text_hidden_states[-1], torch.Tensor)
            ):
                talker_hidden_steps.append(text_hidden_states[-1][:, -1:])

        if len(talker_code_steps) == 0 or len(talker_hidden_steps) == 0:
            raise RuntimeError(
                "Talker generation returned empty hidden/code states; cannot decode."
            )

        talker_codes = torch.stack(talker_code_steps, dim=1)
        talker_hidden_states = torch.cat(talker_hidden_steps, dim=1)[:, :-1]

        first_codebook = talker_codes[0, :, 0]
        stop_positions = torch.nonzero(
            first_codebook == self.config.talker_config.codec_eos_token_id,
            as_tuple=False,
        )
        if stop_positions.numel() > 0:
            effective_length = int(stop_positions[0, 0].item())
        else:
            effective_length = int(talker_codes.shape[1])

        return (
            talker_codes[0, :effective_length],
            talker_hidden_states[0, :effective_length, :],
        )

    def _build_talker_suppress_tokens(self) -> list[int]:
        return [
            token_id
            for token_id in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size,
            )
            if token_id not in (self.config.talker_config.codec_eos_token_id,)
        ]

    def _append_instruct_embed_block(
        self, talker_input_embeds: list[torch.Tensor], instruct_id: torch.Tensor | None
    ) -> None:
        if instruct_id is None:
            return
        talker_input_embeds.append(
            self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
        )

    def _prepare_standard_single_generation(
        self,
        input_id: torch.Tensor,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            talker_input_embed,
            codec_input_embedding,
            tts_eos_embed,
            tts_pad_embed,
        ) = self._build_talker_prefix_embeddings(
            input_id=input_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
        )
        talker_input_embed, trailing_text_hidden = (
            self._append_standard_talker_body_embeddings(
                input_id=input_id,
                talker_input_embed=talker_input_embed,
                codec_input_embedding=codec_input_embedding,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                non_streaming_mode=non_streaming_mode,
            )
        )
        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _prepare_voice_clone_single_generation(
        self,
        input_id: torch.Tensor,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
        ref_code: torch.Tensor | None,
        ref_id: torch.Tensor | None,
        use_icl_prompt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            talker_input_embed,
            codec_input_embedding,
            tts_eos_embed,
            tts_pad_embed,
        ) = self._build_talker_prefix_embeddings(
            input_id=input_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
        )
        talker_input_embed, trailing_text_hidden = (
            self._append_voice_clone_talker_body_embeddings(
                input_id=input_id,
                talker_input_embed=talker_input_embed,
                codec_input_embedding=codec_input_embedding,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                non_streaming_mode=non_streaming_mode,
                ref_code=ref_code,
                ref_id=ref_id,
                use_icl_prompt=use_icl_prompt,
            )
        )
        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _generate_single_with_prepared_prompt(
        self,
        instruct_id: torch.Tensor | None,
        talker_input_embed: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embeds: list[torch.Tensor] = []
        self._append_instruct_embed_block(talker_input_embeds, instruct_id)
        talker_input_embeds.append(talker_input_embed)

        return self._run_talker_generation(
            talker_input_embeds=talker_input_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            suppress_tokens=self._build_talker_suppress_tokens(),
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

    def _generate_voice_design_from_ids(
        self,
        input_id: torch.Tensor,
        instruct_id: torch.Tensor | None,
        language_id: int | None,
        non_streaming_mode: bool,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embed, trailing_text_hidden, tts_pad_embed = (
            self._prepare_standard_single_generation(
                input_id=input_id,
                language_id=language_id,
                speaker_embed=None,
                non_streaming_mode=non_streaming_mode,
            )
        )
        return self._generate_single_with_prepared_prompt(
            instruct_id=instruct_id,
            talker_input_embed=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
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

    def _generate_custom_voice_from_ids(
        self,
        input_id: torch.Tensor,
        instruct_id: torch.Tensor | None,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embed, trailing_text_hidden, tts_pad_embed = (
            self._prepare_standard_single_generation(
                input_id=input_id,
                language_id=language_id,
                speaker_embed=speaker_embed,
                non_streaming_mode=non_streaming_mode,
            )
        )
        return self._generate_single_with_prepared_prompt(
            instruct_id=instruct_id,
            talker_input_embed=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
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

    def _generate_voice_clone_from_ids(
        self,
        input_id: torch.Tensor,
        instruct_id: torch.Tensor | None,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
        ref_code: torch.Tensor | None,
        ref_id: torch.Tensor | None,
        use_icl_prompt: bool,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embed, trailing_text_hidden, tts_pad_embed = (
            self._prepare_voice_clone_single_generation(
                input_id=input_id,
                language_id=language_id,
                speaker_embed=speaker_embed,
                non_streaming_mode=non_streaming_mode,
                ref_code=ref_code,
                ref_id=ref_id,
                use_icl_prompt=use_icl_prompt,
            )
        )
        return self._generate_single_with_prepared_prompt(
            instruct_id=instruct_id,
            talker_input_embed=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
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
    "Qwen3TTSGenerationSingleMixin",
]
