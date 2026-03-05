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
"""Public exports for Qwen3 TTS conditional-generation model classes."""

from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_core import Qwen3TTSPreTrainedModel, mel_spectrogram
from .modeling_qwen3_tts_custom_voice import Qwen3TTSCustomVoiceForConditionalGeneration
from .modeling_qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerModel,
)
from .modeling_qwen3_tts_voice_clone import Qwen3TTSVoiceCloneForConditionalGeneration
from .modeling_qwen3_tts_voice_design import Qwen3TTSVoiceDesignForConditionalGeneration

__all__ = [
    "Qwen3TTSConditionalGenerationBase",
    "Qwen3TTSVoiceDesignForConditionalGeneration",
    "Qwen3TTSCustomVoiceForConditionalGeneration",
    "Qwen3TTSVoiceCloneForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerModel",
    "mel_spectrogram",
]
