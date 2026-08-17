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
from ..modeling_qwen3_tts_types import (
    SubTalkerConfiguration,
    VoiceClonePromptSingle,
)
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
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
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
            subtalker_configuration=subtalker_configuration,
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

        resolved_eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.config.talker_config.codec_eos_token_id
        )
        first_codebook = talker_codes[0, :, 0]
        stop_positions = torch.nonzero(
            first_codebook == resolved_eos_token_id,
            as_tuple=False,
        )
        if stop_positions.numel() > 0:
            effective_length = int(stop_positions[0, 0].item())
        else:
            effective_length = int(talker_codes.shape[1])

        sequences = getattr(talker_result, "sequences", None)
        terminated = (
            isinstance(sequences, torch.Tensor)
            and sequences.numel() != 0
            and int(sequences[0, -1].item()) == resolved_eos_token_id
        )
        return (
            talker_codes[0, :effective_length],
            talker_hidden_states[0, :effective_length, :],
            terminated,
        )

    def _append_instruct_embed_block(
        self, talker_input_embeds: list[torch.Tensor], instruct_id: torch.Tensor | None
    ) -> None:
        if instruct_id is None:
            return
        talker_input_embeds.append(
            self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
        )

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
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embeds: list[torch.Tensor] = []
        self._append_instruct_embed_block(talker_input_embeds, instruct_id)
        talker_input_embeds.append(talker_input_embed)

        talker_codes, talker_hidden_states, _ = self._run_talker_generation(
            talker_input_embeds=talker_input_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            suppress_tokens=self._build_talker_suppress_tokens(eos_token_id),
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
        return talker_codes, talker_hidden_states

    def _generate_turns_with_prepared_prompts(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
        prepared_prompts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not (len(input_ids) == len(instruct_ids) == len(prepared_prompts) != 0):
            raise ValueError("Structured generation turn counts must match.")
        if not output_hidden_states or not return_dict_in_generate:
            raise ValueError(
                "Structured generation requires `output_hidden_states=True` and "
                "`return_dict_in_generate=True`."
            )

        history_embeddings: list[torch.Tensor] = []
        talker_codes_list: list[torch.Tensor] = []
        talker_hidden_states_list: list[torch.Tensor] = []
        for turn_index, (
            input_id,
            instruct_id,
            (talker_input_embed, trailing_text_hidden, tts_pad_embed),
        ) in enumerate(zip(input_ids, instruct_ids, prepared_prompts)):
            turn_embeddings: list[torch.Tensor] = []
            self._append_instruct_embed_block(turn_embeddings, instruct_id)
            turn_embeddings.append(talker_input_embed)

            talker_codes, talker_hidden_states, terminated = (
                self._run_talker_generation(
                    talker_input_embeds=history_embeddings + turn_embeddings,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    suppress_tokens=self._build_talker_suppress_tokens(eos_token_id),
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
            )
            talker_codes_list.append(talker_codes)
            talker_hidden_states_list.append(talker_hidden_states)

            if turn_index + 1 < len(prepared_prompts):
                if not terminated:
                    raise RuntimeError(
                        f"Turn {turn_index + 1} did not reach codec EOS. Increase "
                        "`max_new_tokens` before generating a subsequent turn."
                    )
                history_embeddings.extend(turn_embeddings)
                history_embeddings.append(
                    self._build_generated_codec_history_embeddings(
                        talker_codes=talker_codes,
                        trailing_text_hidden=trailing_text_hidden,
                        tts_pad_embed=tts_pad_embed,
                    )
                )
                history_embeddings.append(
                    self._build_codec_eos_history_embedding(
                        tts_pad_embed=tts_pad_embed,
                        trailing_text_hidden=trailing_text_hidden,
                        generated_length=int(talker_codes.shape[0]),
                        input_dtype=input_id.dtype,
                        eos_token_id=(
                            eos_token_id
                            if eos_token_id is not None
                            else self.config.talker_config.codec_eos_token_id
                        ),
                    )
                )

        return talker_codes_list, talker_hidden_states_list

    def _generate_standard_turns_from_ids(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
        non_streaming_mode: bool,
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prepared_prompts = [
            self._prepare_standard_generation(
                input_id=input_id,
                language_id=language_id,
                speaker_embed=speaker_embed,
                non_streaming_mode=non_streaming_mode,
            )
            for input_id in input_ids
        ]
        return self._generate_turns_with_prepared_prompts(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            prepared_prompts=prepared_prompts,
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

    def _generate_voice_clone_turns_from_ids(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor | None],
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
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prepared_prompts = [
            self._prepare_voice_clone_generation(
                input_id=input_id,
                language_id=language_id,
                speaker_embed=speaker_embed,
                non_streaming_mode=non_streaming_mode,
                ref_code=ref_code if turn_index == 0 else None,
                ref_id=ref_id if turn_index == 0 else None,
                use_icl_prompt=use_icl_prompt and turn_index == 0,
            )
            for turn_index, input_id in enumerate(input_ids)
        ]
        return self._generate_turns_with_prepared_prompts(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            prepared_prompts=prepared_prompts,
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

    def _generate_standard_from_ids(
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
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embed, trailing_text_hidden, tts_pad_embed = (
            self._prepare_standard_generation(
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
            subtalker_configuration=subtalker_configuration,
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
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        output_hidden_states: bool,
        return_dict_in_generate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        talker_input_embed, trailing_text_hidden, tts_pad_embed = (
            self._prepare_voice_clone_generation(
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
            subtalker_configuration=subtalker_configuration,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )


__all__ = [
    "Qwen3TTSGenerationSingleMixin",
]
