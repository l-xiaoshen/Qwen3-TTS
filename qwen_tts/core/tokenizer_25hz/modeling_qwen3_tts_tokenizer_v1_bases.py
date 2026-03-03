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
"""Shared pretrained base classes for Qwen3TTSTokenizerV1 splits."""

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring

from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1DecoderConfig,
    Qwen3TTSTokenizerV1EncoderConfig,
)


@auto_docstring
class Qwen3TTSTokenizerV1DecoderPreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring
class Qwen3TTSTokenizerV1EncoderPreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1EncoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
