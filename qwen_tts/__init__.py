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

"""
qwen_tts: Qwen-TTS package.
"""

from .core import SpeakerConfiguration
from .inference.qwen3_tts_base_model import Qwen3TTSBaseModel
from .inference.qwen3_tts_custom_voice_model import Qwen3TTSCustomVoiceModel
from .inference.qwen3_tts_voice_clone_model import (
    Qwen3TTSVoiceCloneModel,
    VoiceClonePromptItem,
)
from .inference.qwen3_tts_voice_design_model import Qwen3TTSVoiceDesignModel
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = [
    "Qwen3TTSBaseModel",
    "SpeakerConfiguration",
    "Qwen3TTSCustomVoiceModel",
    "Qwen3TTSVoiceCloneModel",
    "Qwen3TTSVoiceDesignModel",
    "VoiceClonePromptItem",
    "Qwen3TTSTokenizer",
]
