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
"""Composed generation mixin for Qwen3 TTS."""

from .modeling_qwen3_tts_generation_embeddings import (
    Qwen3TTSGenerationEmbeddingsMixin,
)
from .modeling_qwen3_tts_generation_runner import (
    Qwen3TTSGenerationRunnerMixin,
)
from .modeling_qwen3_tts_generation_validation import (
    Qwen3TTSGenerationValidationMixin,
)


class Qwen3TTSGenerationMixin(
    Qwen3TTSGenerationValidationMixin,
    Qwen3TTSGenerationEmbeddingsMixin,
    Qwen3TTSGenerationRunnerMixin,
):
    """Aggregate all Qwen3 TTS generation helpers."""


__all__ = [
    "Qwen3TTSGenerationMixin",
]
