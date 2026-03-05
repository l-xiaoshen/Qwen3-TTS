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
"""Batch-oriented generation orchestration helpers."""

from collections.abc import Sequence
from typing import Optional

import torch

from ..configuration_qwen3_tts import Qwen3TTSConfig
from ..modeling_qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
from ..modeling_qwen3_tts_types import VoiceClonePrompt
from .core import Qwen3TTSGenerationCoreMixin


class Qwen3TTSGenerationBatchMixin(Qwen3TTSGenerationCoreMixin):
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

    def _resolve_voice_clone_speaker_embed_batch(
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

    def _run_talker_generation_batch(
        self,
        talker_input_embeds: list[list[torch.Tensor]],
        trailing_text_hiddens: list[torch.Tensor],
        tts_pad_embed_last: torch.Tensor,
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
        eos_token_id: Optional[int],
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        merged_talker_input_embeds: list[torch.Tensor] = []
        for sample_embeds in talker_input_embeds:
            if len(sample_embeds) == 0:
                raise RuntimeError(
                    "Each batch item must contain at least one embed block."
                )
            merged_talker_input_embeds.append(torch.cat(sample_embeds, dim=1))

        original_lengths = torch.tensor(
            [embed.shape[1] for embed in merged_talker_input_embeds]
        )
        sequences = [embed.squeeze(0) for embed in merged_talker_input_embeds]
        sequences_reversed = [embed.flip(dims=[0]) for embed in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed, batch_first=True, padding_value=0.0
        )
        talker_input_embeds_tensor = padded_reversed.flip(dims=[1])

        generated_batch_size, max_len = (
            talker_input_embeds_tensor.shape[0],
            talker_input_embeds_tensor.shape[1],
        )
        indices = torch.arange(max_len).expand(generated_batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (
            (indices >= num_pads.unsqueeze(1))
            .long()
            .to(talker_input_embeds_tensor.device)
        )

        pad_embedding_vector = tts_pad_embed_last.squeeze()
        sequences_to_pad = [hidden.squeeze(0) for hidden in trailing_text_hiddens]
        trailing_text_original_lengths = [
            hidden.shape[0] for hidden in sequences_to_pad
        ]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad, batch_first=True, padding_value=0.0
        )
        arange_tensor = torch.arange(
            max(trailing_text_original_lengths), device=padded_hiddens.device
        ).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(
            trailing_text_original_lengths, device=padded_hiddens.device
        ).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens_tensor = padded_hiddens

        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds_tensor,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens_tensor,
            tts_pad_embed=tts_pad_embed_last,
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

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(
            has_stop_token, stop_indices, talker_codes.shape[1]
        )

        talker_codes_list = [
            talker_codes[index, :length]
            for index, length in enumerate(effective_lengths)
        ]
        talker_hidden_states_list = [
            talker_hidden_states[index, :length, :]
            for index, length in enumerate(effective_lengths)
        ]
        return talker_codes_list, talker_hidden_states_list

    def _build_talker_suppress_tokens_batch(self) -> list[int]:
        return [
            token_id
            for token_id in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size,
            )
            if token_id not in (self.config.talker_config.codec_eos_token_id,)
        ]

    def _append_instruct_embed_blocks_batch(
        self,
        talker_input_embeds: list[list[torch.Tensor]],
        instruct_ids: list[torch.Tensor | None],
    ) -> None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(instruct_id)
                    )
                )

    def _prepare_standard_batch_sample(
        self,
        input_id: torch.Tensor,
        language: str,
        speaker: str | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_id = self._resolve_language_id(language, speaker)
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

    def _prepare_voice_clone_batch_sample(
        self,
        input_id: torch.Tensor,
        language: str,
        speaker: str | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
        ref_code: torch.Tensor | None,
        ref_id: torch.Tensor | None,
        use_icl_prompt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_id = self._resolve_language_id(language, speaker)
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

    def _generate_batch_with_prepared_prompts(
        self,
        instruct_ids: list[torch.Tensor | None],
        prepared_talker_input_embeds: list[torch.Tensor],
        trailing_text_hiddens: list[torch.Tensor],
        tts_pad_embed_last: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_dosample: bool,
        subtalker_top_k: int,
        subtalker_top_p: float,
        subtalker_temperature: float,
        eos_token_id: Optional[int],
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not (
            len(prepared_talker_input_embeds)
            == len(trailing_text_hiddens)
            == len(instruct_ids)
        ):
            raise ValueError(
                "Batch size mismatch in prepared prompt embeddings and generation inputs."
            )

        talker_input_embeds: list[list[torch.Tensor]] = [
            [] for _ in range(len(prepared_talker_input_embeds))
        ]
        self._append_instruct_embed_blocks_batch(talker_input_embeds, instruct_ids)
        for index, talker_input_embed in enumerate(prepared_talker_input_embeds):
            talker_input_embeds[index].append(talker_input_embed)

        return self._run_talker_generation_batch(
            talker_input_embeds=talker_input_embeds,
            trailing_text_hiddens=trailing_text_hiddens,
            tts_pad_embed_last=tts_pad_embed_last,
            suppress_tokens=self._build_talker_suppress_tokens_batch(),
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

    def _generate_voice_design_batch_from_ids(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
        languages: list[str],
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
        eos_token_id: Optional[int],
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not (len(input_ids) == len(instruct_ids) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: input_ids={len(input_ids)}, instruct_ids={len(instruct_ids)}, languages={len(languages)}"
            )

        prepared_talker_input_embeds: list[torch.Tensor] = []
        trailing_text_hiddens: list[torch.Tensor] = []
        tts_pad_embed_last: torch.Tensor | None = None
        for input_id, language in zip(input_ids, languages):
            talker_input_embed, trailing_text_hidden, tts_pad_embed = (
                self._prepare_standard_batch_sample(
                    input_id=input_id,
                    language=language,
                    speaker=None,
                    speaker_embed=None,
                    non_streaming_mode=non_streaming_mode,
                )
            )
            prepared_talker_input_embeds.append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
            tts_pad_embed_last = tts_pad_embed

        if tts_pad_embed_last is None:
            raise RuntimeError("`tts_pad_embed` was not created during generation.")

        return self._generate_batch_with_prepared_prompts(
            instruct_ids=instruct_ids,
            prepared_talker_input_embeds=prepared_talker_input_embeds,
            trailing_text_hiddens=trailing_text_hiddens,
            tts_pad_embed_last=tts_pad_embed_last,
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

    def _generate_custom_voice_batch_from_ids(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
        languages: list[str],
        speakers: list[str | None],
        speaker_embeds: list[torch.Tensor | None],
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
        eos_token_id: Optional[int],
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not (
            len(input_ids)
            == len(instruct_ids)
            == len(languages)
            == len(speakers)
            == len(speaker_embeds)
        ):
            raise ValueError("Batch size mismatch in custom-voice generation inputs.")

        prepared_talker_input_embeds: list[torch.Tensor] = []
        trailing_text_hiddens: list[torch.Tensor] = []
        tts_pad_embed_last: torch.Tensor | None = None
        for input_id, language, speaker, speaker_embed in zip(
            input_ids, languages, speakers, speaker_embeds
        ):
            talker_input_embed, trailing_text_hidden, tts_pad_embed = (
                self._prepare_standard_batch_sample(
                    input_id=input_id,
                    language=language,
                    speaker=speaker,
                    speaker_embed=speaker_embed,
                    non_streaming_mode=non_streaming_mode,
                )
            )
            prepared_talker_input_embeds.append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
            tts_pad_embed_last = tts_pad_embed

        if tts_pad_embed_last is None:
            raise RuntimeError("`tts_pad_embed` was not created during generation.")

        return self._generate_batch_with_prepared_prompts(
            instruct_ids=instruct_ids,
            prepared_talker_input_embeds=prepared_talker_input_embeds,
            trailing_text_hiddens=trailing_text_hiddens,
            tts_pad_embed_last=tts_pad_embed_last,
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

    def _generate_voice_clone_batch_from_ids(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
        languages: list[str],
        speakers: list[str | None],
        speaker_embeds: list[torch.Tensor | None],
        ref_codes: list[torch.Tensor | None],
        ref_ids: list[torch.Tensor | None],
        use_icl_prompts: list[bool],
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
        eos_token_id: Optional[int],
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not (
            len(input_ids)
            == len(instruct_ids)
            == len(languages)
            == len(speakers)
            == len(speaker_embeds)
            == len(ref_codes)
            == len(ref_ids)
            == len(use_icl_prompts)
        ):
            raise ValueError("Batch size mismatch in voice-clone generation inputs.")

        prepared_talker_input_embeds: list[torch.Tensor] = []
        trailing_text_hiddens: list[torch.Tensor] = []
        tts_pad_embed_last: torch.Tensor | None = None
        for (
            input_id,
            language,
            speaker,
            speaker_embed,
            ref_code,
            ref_id,
            use_icl_prompt,
        ) in zip(
            input_ids,
            languages,
            speakers,
            speaker_embeds,
            ref_codes,
            ref_ids,
            use_icl_prompts,
        ):
            talker_input_embed, trailing_text_hidden, tts_pad_embed = (
                self._prepare_voice_clone_batch_sample(
                    input_id=input_id,
                    language=language,
                    speaker=speaker,
                    speaker_embed=speaker_embed,
                    non_streaming_mode=non_streaming_mode,
                    ref_code=ref_code,
                    ref_id=ref_id,
                    use_icl_prompt=use_icl_prompt,
                )
            )
            prepared_talker_input_embeds.append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
            tts_pad_embed_last = tts_pad_embed

        if tts_pad_embed_last is None:
            raise RuntimeError("`tts_pad_embed` was not created during generation.")

        return self._generate_batch_with_prepared_prompts(
            instruct_ids=instruct_ids,
            prepared_talker_input_embeds=prepared_talker_input_embeds,
            trailing_text_hiddens=trailing_text_hiddens,
            tts_pad_embed_last=tts_pad_embed_last,
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
    "Qwen3TTSGenerationBatchMixin",
]
