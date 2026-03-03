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
"""Talker exports for split Qwen3TTS modeling modules."""

from .modeling_qwen3_tts_talker_model import (
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerModel,
)

__all__ = [
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSTalkerModel",
]
