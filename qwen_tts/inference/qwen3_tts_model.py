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
Re-exports for Qwen3 TTS inference: standalone feature classes and loader.
"""
from .base import AudioLike, MaybeList, Qwen3TTSModelBase
from .custom_voice import Qwen3TTSCustomVoice
from .voice_clone import Qwen3TTSVoiceClone
from .voice_clone_prompt_item import VoiceClonePromptItem
from .voice_design import Qwen3TTSVoiceDesign


def load_qwen3_tts(pretrained_model_name_or_path: str, **kwargs):
    """
    Load a Qwen3 TTS model from a checkpoint and return the appropriate wrapper class
    (Qwen3TTSCustomVoice, Qwen3TTSVoiceDesign, or Qwen3TTSVoiceClone) based on
    the model's tts_model_type. Use this when you have a path but do not know the type.
    """
    tmp = Qwen3TTSVoiceClone.from_pretrained(pretrained_model_name_or_path, **kwargs)
    mt = tmp.model.tts_model_type
    if mt == "custom_voice":
        return Qwen3TTSCustomVoice(
            model=tmp.model,
            processor=tmp.processor,
            generate_defaults=tmp.generate_defaults,
        )
    if mt == "voice_design":
        return Qwen3TTSVoiceDesign(
            model=tmp.model,
            processor=tmp.processor,
            generate_defaults=tmp.generate_defaults,
        )
    if mt == "base":
        return tmp
    raise ValueError(f"Unknown tts_model_type: {mt}")


__all__ = [
    "Qwen3TTSVoiceClone",
    "Qwen3TTSVoiceDesign",
    "Qwen3TTSCustomVoice",
    "Qwen3TTSModelBase",
    "VoiceClonePromptItem",
    "AudioLike",
    "MaybeList",
    "load_qwen3_tts",
]
