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


class SubTalkerConfiguration(TypedDict, total=False):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float


GenerateConfigPrimitive = str | int | float | bool | None
GenerateConfigValue = (
    GenerateConfigPrimitive
    | list["GenerateConfigValue"]
    | dict[str, "GenerateConfigValue"]
)


__all__ = [
    "VoiceClonePrompt",
    "VoiceClonePromptSingle",
    "SubTalkerConfiguration",
    "GenerateConfigPrimitive",
    "GenerateConfigValue",
]
