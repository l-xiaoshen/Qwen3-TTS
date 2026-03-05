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
"""Embedding assembly helpers for Qwen3 TTS generation."""

import torch

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration


class Qwen3TTSGenerationEmbeddingsMixin:
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
        raise NotImplementedError

    def _build_talker_prefix_embeddings(
        self,
        input_id: torch.Tensor,
        language_id: int | None,
        speaker_embed: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                    dtype=input_id.dtype,
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
                dtype=input_id.dtype,
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
                dtype=input_id.dtype,
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
            self.talker.get_text_embeddings()(input_id[:, :3])
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

    def _append_talker_body_embeddings(
        self,
        input_id: torch.Tensor,
        talker_input_embed: torch.Tensor,
        codec_input_embedding: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        non_streaming_mode: bool,
        ref_code: torch.Tensor | None,
        ref_id: torch.Tensor | None,
        use_icl_prompt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_icl_prompt and ref_code is not None:
            if ref_id is None:
                raise ValueError(
                    "`ref_ids` is required for ICL mode voice clone generation."
                )
            icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=ref_code.to(self.talker.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            return talker_input_embed, trailing_text_hidden

        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                self.talker.text_projection(
                    self.talker.get_text_embeddings()(input_id[:, 3:4])
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
                                self.talker.get_text_embeddings()(input_id[:, 3:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
                    + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [
                                [self.config.talker_config.codec_pad_id]
                                * (input_id[:, 3:-5].shape[1] + 1)
                            ],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    ),
                    tts_pad_embed
                    + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[self.config.talker_config.codec_bos_id]],
                            device=self.talker.device,
                            dtype=input_id.dtype,
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
                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                    ),
                    tts_eos_embed,
                ),
                dim=1,
            )
        return talker_input_embed, trailing_text_hidden


__all__ = [
    "Qwen3TTSGenerationEmbeddingsMixin",
]
