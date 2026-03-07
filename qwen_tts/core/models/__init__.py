# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
from .configuration_qwen3_tts import Qwen3TTSConfig
from ..config import SpeakerConfiguration
from .modeling_qwen3_tts_base import Qwen3TTSConditionalGenerationBase
from .modeling_qwen3_tts_core import mel_spectrogram
from .modeling_qwen3_tts_custom_voice import Qwen3TTSCustomVoiceForConditionalGeneration
from .modeling_qwen3_tts_types import SubTalkerConfiguration
from .modeling_qwen3_tts_voice_clone import Qwen3TTSVoiceCloneForConditionalGeneration
from .modeling_qwen3_tts_voice_design import Qwen3TTSVoiceDesignForConditionalGeneration
from .processing_qwen3_tts import Qwen3TTSProcessor

__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSConditionalGenerationBase",
    "SpeakerConfiguration",
    "SubTalkerConfiguration",
    "Qwen3TTSVoiceDesignForConditionalGeneration",
    "Qwen3TTSCustomVoiceForConditionalGeneration",
    "Qwen3TTSVoiceCloneForConditionalGeneration",
    "Qwen3TTSProcessor",
    "mel_spectrogram",
]
