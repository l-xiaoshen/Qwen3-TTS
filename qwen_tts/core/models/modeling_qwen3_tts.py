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
    "Qwen3TTSCustomVoiceForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSVoiceCloneForConditionalGeneration",
    "Qwen3TTSVoiceDesignForConditionalGeneration",
    "mel_spectrogram",
]
