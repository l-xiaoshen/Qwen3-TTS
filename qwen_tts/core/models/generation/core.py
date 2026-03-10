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
"""Core generation helpers independent from batch orchestration."""

import torch

from ...config import SpeakerConfiguration
from ..configuration_qwen3_tts import Qwen3TTSConfig
from ..modeling_qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration


class Qwen3TTSGenerationCoreMixin:
    config: Qwen3TTSConfig
    talker: Qwen3TTSTalkerForConditionalGeneration

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(
                    self.talker.code_predictor.get_input_embeddings()[i - 1](
                        ref_code[:, i : i + 1]
                    )
                )
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.talker.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.talker_config.codec_bos_id]],
                        device=self.talker.device,
                        dtype=text_id.dtype,
                    )
                ),
                codec_embed,
            ],
            dim=1,
        )

        # compute lengths
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat(
                [icl_input_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed

        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]

        text_embed = torch.cat(
            [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
        )
        return text_embed + codec_embed, tts_pad_embed

    @staticmethod
    def _require_non_empty_text_ids(
        text_id: torch.Tensor, prompt_name: str
    ) -> torch.Tensor:
        if text_id.shape[1] == 0:
            raise ValueError(
                f"`{prompt_name}` must contain at least one content token."
            )
        return text_id

    def _build_talker_prefix_embeddings(
        self,
        input_role_id: torch.Tensor,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = input_role_id.dtype
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [
                        [
                            self.config.tts_bos_token_id,
                            self.config.tts_eos_token_id,
                            self.config.tts_pad_token_id,
                        ]
                    ],
                    device=self.talker.device,
                    dtype=input_dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill_list = [
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill_list = [
                [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    language_id,
                    self.config.talker_config.codec_think_eos_id,
                ]
            ]

        codec_input_embedding_0 = self.talker.get_input_embeddings()(
            torch.tensor(
                codec_prefill_list,
                device=self.talker.device,
                dtype=input_dtype,
            )
        )
        codec_input_embedding_1 = self.talker.get_input_embeddings()(
            torch.tensor(
                [
                    [
                        self.config.talker_config.codec_pad_id,
                        self.config.talker_config.codec_bos_id,
                    ]
                ],
                device=self.talker.device,
                dtype=input_dtype,
            )
        )
        if speaker_embed is None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat(
                [
                    codec_input_embedding_0,
                    speaker_embed.view(1, 1, -1),
                    codec_input_embedding_1,
                ],
                dim=1,
            )

        talker_input_embed_role = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_role_id)
        )
        talker_input_embed_prefill = (
            torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            )
            + codec_input_embedding[:, :-1]
        )
        talker_input_embed = torch.cat(
            (talker_input_embed_role, talker_input_embed_prefill), dim=1
        )
        return talker_input_embed, codec_input_embedding, tts_eos_embed, tts_pad_embed

    def _append_icl_talker_body_embeddings(
        self,
        input_text_id: torch.Tensor,
        talker_input_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        non_streaming_mode: bool,
        ref_code: torch.Tensor,
        ref_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ref_id is None:
            raise ValueError(
                "`ref_id` is required for ICL mode voice clone generation."
            )
        input_text_id = self._require_non_empty_text_ids(input_text_id, "input_text_id")
        ref_id = self._require_non_empty_text_ids(ref_id, "ref_id")
        icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
            text_id=input_text_id,
            ref_id=ref_id,
            ref_code=ref_code.to(self.talker.device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        return talker_input_embed, trailing_text_hidden

    def _append_standard_talker_body_embeddings(
        self,
        input_text_id: torch.Tensor,
        talker_input_embed: torch.Tensor,
        codec_input_embedding: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_text_id = self._require_non_empty_text_ids(input_text_id, "input_text_id")
        input_dtype = input_text_id.dtype

        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                self.talker.text_projection(
                    self.talker.get_text_embeddings()(input_text_id[:, :1])
                )
                + codec_input_embedding[:, -1:],
            ],
            dim=1,
        )
        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat(
                        (
                            self.talker.text_projection(
                                self.talker.get_text_embeddings()(input_text_id)
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
                    + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [
                                [self.config.talker_config.codec_pad_id]
                                * (input_text_id.shape[1] + 1)
                            ],
                            device=self.talker.device,
                            dtype=input_dtype,
                        )
                    ),
                    tts_pad_embed
                    + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[self.config.talker_config.codec_bos_id]],
                            device=self.talker.device,
                            dtype=input_dtype,
                        )
                    ),
                ],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            trailing_text_hidden = torch.cat(
                (
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_text_id[:, 1:])
                    ),
                    tts_eos_embed,
                ),
                dim=1,
            )
        return talker_input_embed, trailing_text_hidden

    def _append_voice_clone_talker_body_embeddings(
        self,
        input_text_id: torch.Tensor,
        talker_input_embed: torch.Tensor,
        codec_input_embedding: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        non_streaming_mode: bool,
        ref_code: torch.Tensor | None,
        ref_id: torch.Tensor | None,
        use_icl_prompt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_icl_prompt:
            if ref_code is None:
                raise ValueError(
                    "`ref_code` is required for ICL mode voice clone generation."
                )
            return self._append_icl_talker_body_embeddings(
                input_text_id=input_text_id,
                talker_input_embed=talker_input_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                non_streaming_mode=non_streaming_mode,
                ref_code=ref_code,
                ref_id=ref_id,
            )
        return self._append_standard_talker_body_embeddings(
            input_text_id=input_text_id,
            talker_input_embed=talker_input_embed,
            codec_input_embedding=codec_input_embedding,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            non_streaming_mode=non_streaming_mode,
        )

    def _prepare_standard_generation(
        self,
        input_role_id: torch.Tensor,
        input_text_id: torch.Tensor,
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
            input_role_id=input_role_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
        )
        talker_input_embed, trailing_text_hidden = (
            self._append_standard_talker_body_embeddings(
                input_text_id=input_text_id,
                talker_input_embed=talker_input_embed,
                codec_input_embedding=codec_input_embedding,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                non_streaming_mode=non_streaming_mode,
            )
        )
        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _prepare_voice_clone_generation(
        self,
        input_role_id: torch.Tensor,
        input_text_id: torch.Tensor,
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
            input_role_id=input_role_id,
            language_id=language_id,
            speaker_embed=speaker_embed,
        )
        talker_input_embed, trailing_text_hidden = (
            self._append_voice_clone_talker_body_embeddings(
                input_text_id=input_text_id,
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

    def _build_talker_suppress_tokens(self) -> list[int]:
        return [
            token_id
            for token_id in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size,
            )
            if token_id not in (self.config.talker_config.codec_eos_token_id,)
        ]

    def _resolve_custom_voice_speaker_embed(
        self, speaker: SpeakerConfiguration, input_dtype: torch.dtype
    ) -> torch.Tensor | None:
        if len(speaker) == 0:
            return None

        spk_id_map = self.config.talker_config.spk_id
        if spk_id_map is None:
            raise ValueError("Speaker map is not available in config")

        mixed_embed: torch.Tensor | None = None
        supported = sorted(list(spk_id_map.keys()))
        for speaker_name, weight in speaker.items():
            speaker_lower = speaker_name.lower()
            if speaker_lower not in spk_id_map:
                raise ValueError(
                    f"Unsupported speaker: {speaker_name}. Supported speakers: {supported}"
                )

            speaker_embed = self.talker.get_input_embeddings()(
                torch.tensor(
                    spk_id_map[speaker_lower],
                    device=self.talker.device,
                    dtype=input_dtype,
                )
            )
            weighted_embed = speaker_embed * speaker_embed.new_tensor(float(weight))
            mixed_embed = (
                weighted_embed if mixed_embed is None else mixed_embed + weighted_embed
            )

        return mixed_embed

    def _resolve_language_id(self, language: str, speaker: str) -> int | None:
        language_lower = language.lower()
        language_map = self.config.talker_config.codec_language_id
        if language_map is None:
            raise ValueError("Language map is not available in config")
        if language_lower == "auto":
            language_id = None
        else:
            if language_lower not in language_map:
                supported = sorted(list(language_map.keys()))
                raise ValueError(
                    f"Unsupported language: {language}. Supported languages: {supported}"
                )
            language_id = language_map[language_lower]

        dialect_map = self.config.talker_config.spk_is_dialect or {}
        dialect_value = (
            dialect_map.get(speaker.lower(), False) if speaker != "" else False
        )
        if (
            language_lower in ["chinese", "auto"]
            and speaker != ""
            and dialect_value is not False
            and isinstance(dialect_value, str)
            and dialect_value in language_map
        ):
            language_id = language_map[dialect_value]
        return language_id

    def _resolve_speaker_language_bucket(
        self, speaker: str, language: str
    ) -> str | None:
        language_lower = language.lower()
        if language_lower not in ["chinese", "auto"] or speaker == "":
            return None

        language_map = self.config.talker_config.codec_language_id
        if language_map is None:
            raise ValueError("Language map is not available in config")

        dialect_map = self.config.talker_config.spk_is_dialect or {}
        dialect_value = dialect_map.get(speaker.lower(), False)
        if isinstance(dialect_value, str) and dialect_value in language_map:
            return dialect_value
        return None

    def _resolve_language_id_for_speaker_config(
        self, language: str, speaker: SpeakerConfiguration
    ) -> int | None:
        language_id = self._resolve_language_id(language, "")
        language_lower = language.lower()
        if len(speaker) == 0 or language_lower not in ["chinese", "auto"]:
            return language_id

        language_map = self.config.talker_config.codec_language_id
        if language_map is None:
            raise ValueError("Language map is not available in config")
        active_speakers = [
            speaker_name
            for speaker_name, weight in speaker.items()
            if float(weight) != 0.0
        ]
        if len(active_speakers) == 0:
            return language_id

        first_speaker_bucket = self._resolve_speaker_language_bucket(
            active_speakers[0], language
        )
        for speaker_name in active_speakers[1:]:
            speaker_bucket = self._resolve_speaker_language_bucket(
                speaker_name, language
            )
            if speaker_bucket != first_speaker_bucket:
                raise ValueError(
                    "All speakers in `speaker` must resolve to the same language bucket."
                )

        if first_speaker_bucket is None:
            return language_id
        return language_map[first_speaker_bucket]


__all__ = [
    "Qwen3TTSGenerationCoreMixin",
]
