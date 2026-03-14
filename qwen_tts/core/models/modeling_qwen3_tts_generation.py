"""Composed generation mixin for Qwen3 TTS."""

from .generation import (
    Qwen3TTSGenerationBatchMixin,
    Qwen3TTSGenerationSingleMixin,
)


class Qwen3TTSGenerationMixin(
    Qwen3TTSGenerationSingleMixin,
    Qwen3TTSGenerationBatchMixin,
):
    """Aggregate all Qwen3 TTS generation helpers."""


__all__ = [
    "Qwen3TTSGenerationMixin",
]
