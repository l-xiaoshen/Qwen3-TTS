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
"""Shared types for Qwen3 TTS conditional generation modules."""

from typing import TypedDict

import torch


class VoiceClonePrompt(TypedDict):
    ref_code: list[torch.Tensor | None]
    ref_spk_embedding: list[torch.Tensor]
    x_vector_only_mode: list[bool]
    icl_mode: list[bool]


class VoiceClonePromptSingle(TypedDict):
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool


GenerateConfigPrimitive = str | int | float | bool | None
GenerateConfigValue = (
    GenerateConfigPrimitive
    | list["GenerateConfigValue"]
    | dict[str, "GenerateConfigValue"]
)


__all__ = [
    "VoiceClonePrompt",
    "VoiceClonePromptSingle",
    "GenerateConfigPrimitive",
    "GenerateConfigValue",
]
