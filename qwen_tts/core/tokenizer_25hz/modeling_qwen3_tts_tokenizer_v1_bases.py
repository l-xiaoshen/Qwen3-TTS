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
