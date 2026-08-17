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

from .core import CodecSpecialTokenIds, SpeakerConfiguration, SubTalkerConfiguration
from .inference.qwen3_tts_base_model import (
    AudioLike,
    Qwen3TTSBaseModel,
    TTSBatchInput,
    TTSInput,
    TTSInputItem,
)
from .inference.qwen3_tts_custom_voice_model import (
    CustomVoicePromptItem,
    Qwen3TTSCustomVoiceModel,
)
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from .inference.qwen3_tts_voice_clone_model import (
    Qwen3TTSVoiceCloneModel,
    VoiceClonePromptItem,
)
from .inference.qwen3_tts_voice_design_model import Qwen3TTSVoiceDesignModel

__all__ = [
    "AudioLike",
    "CodecSpecialTokenIds",
    "CustomVoicePromptItem",
    "Qwen3TTSBaseModel",
    "Qwen3TTSCustomVoiceModel",
    "Qwen3TTSTokenizer",
    "Qwen3TTSVoiceCloneModel",
    "Qwen3TTSVoiceDesignModel",
    "SpeakerConfiguration",
    "SubTalkerConfiguration",
    "TTSBatchInput",
    "TTSInput",
    "TTSInputItem",
    "VoiceClonePromptItem",
]
