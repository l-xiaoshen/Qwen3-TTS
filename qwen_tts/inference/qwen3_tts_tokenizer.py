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
from collections.abc import Callable
from typing import Literal, Protocol, cast, overload, runtime_checkable

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from transformers.models.auto.modeling_auto import AutoModel
from typing_extensions import Self

from ..core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from ..core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2EncoderOutput,
    Qwen3TTSTokenizerV2Model,
)
from ..core.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1Config,
)
from ..core.tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1EncoderOutput,
    Qwen3TTSTokenizerV1Model,
)
from .qwen3_tts_tokenizer_audio_mixin import Qwen3TTSTokenizerAudioMixin
from .qwen3_tts_tokenizer_decode_mixin import Qwen3TTSTokenizerDecodeMixin

AudioInput = (
    str  # wav path, or base64 string
    | np.ndarray  # 1-D float array
    | list[str]
    | list[np.ndarray]
)


@runtime_checkable
class _TokenizerFeatureExtractor(Protocol):
    sampling_rate: int | float

    def __call__(
        self,
        *,
        raw_audio: list[np.ndarray],
        sampling_rate: int,
        return_tensors: str,
    ) -> BatchFeature: ...


TokenizerModel = Qwen3TTSTokenizerV1Model | Qwen3TTSTokenizerV2Model
TokenizerEncoderOutput = (
    Qwen3TTSTokenizerV1EncoderOutput | Qwen3TTSTokenizerV2EncoderOutput
)
TokenizerEncoderTuple = (
    tuple[list[torch.LongTensor]]
    | tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]
)


class Qwen3TTSTokenizer(Qwen3TTSTokenizerDecodeMixin, Qwen3TTSTokenizerAudioMixin):
    """
    A wrapper for Qwen3 TTS Tokenizer 25Hz/12Hz with HuggingFace-style loading.

    - from_pretrained(): loads speech tokenizer model via AutoModel and feature_extractor via AutoFeatureExtractor.
    - encode(): supports wav path(s), base64 audio string(s), numpy array(s).
    - decode(): accepts either the raw model encode output, or a minimal dict/list-of-dicts.

    Notes:
    - For numpy array input, you must pass `sr` so the audio can be resampled to model sample rate.
    - Returned audio is float32 numpy arrays and the output sample rate.
    """

    def __init__(self) -> None:
        self.model: TokenizerModel | None = None
        self.feature_extractor: _TokenizerFeatureExtractor | None = None
        self.config: Qwen3TTSTokenizerV1Config | Qwen3TTSTokenizerV2Config | None = None
        self.device: torch.device | None = None

    def _require_initialized(
        self,
    ) -> tuple[TokenizerModel, _TokenizerFeatureExtractor, torch.device]:
        if self.model is None:
            raise RuntimeError("Tokenizer model is not initialized.")
        if self.feature_extractor is None:
            raise RuntimeError("Tokenizer feature_extractor is not initialized.")
        if self.device is None:
            raise RuntimeError("Tokenizer device is not initialized.")
        return self.model, self.feature_extractor, self.device

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: object
    ) -> Self:
        """
        Initialize tokenizer with HuggingFace `from_pretrained` style.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory.
            **kwargs (object):
                Forwarded to `AutoModel.from_pretrained(...)` directly.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="eager".

        Returns:
            Qwen3TTSTokenizer:
                Initialized instance with `model`, `feature_extractor`, `config`.
        """
        inst = cls()

        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)

        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

        feature_extractor_loader = cast(
            Callable[..., object], AutoFeatureExtractor.from_pretrained
        )
        feature_extractor_raw = feature_extractor_loader(pretrained_model_name_or_path)
        if not isinstance(feature_extractor_raw, _TokenizerFeatureExtractor):
            raise TypeError(
                "AutoFeatureExtractor returned an incompatible feature extractor."
            )
        if not isinstance(feature_extractor_raw.sampling_rate, (int, float)):
            raise TypeError("Feature extractor `sampling_rate` must be numeric.")
        inst.feature_extractor = feature_extractor_raw

        model_loader = cast(Callable[..., object], AutoModel.from_pretrained)
        model_raw = model_loader(pretrained_model_name_or_path, **kwargs)
        if not isinstance(
            model_raw, (Qwen3TTSTokenizerV1Model, Qwen3TTSTokenizerV2Model)
        ):
            raise TypeError(
                "AutoModel returned unexpected tokenizer model type: "
                f"{type(model_raw).__name__}"
            )
        model = model_raw
        if not isinstance(
            model.config, (Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV2Config)
        ):
            raise TypeError("Tokenizer model returned an incompatible config.")
        inst.model = model
        inst.config = model.config

        model_device: object = getattr(model, "device", None)
        if isinstance(model_device, torch.device):
            inst.device = model_device
        else:
            # fallback: infer from first parameter device
            try:
                inst.device = next(model.parameters()).device
            except StopIteration:
                inst.device = torch.device("cpu")

        return inst

    @overload
    def encode(
        self,
        audios: AudioInput,
        sr: int | None = None,
        *,
        return_dict: Literal[True] = True,
    ) -> TokenizerEncoderOutput: ...

    @overload
    def encode(
        self,
        audios: AudioInput,
        sr: int | None = None,
        *,
        return_dict: Literal[False],
    ) -> TokenizerEncoderTuple: ...

    def encode(
        self,
        audios: AudioInput,
        sr: int | None = None,
        *,
        return_dict: bool = True,
    ) -> TokenizerEncoderOutput | TokenizerEncoderTuple:
        """
        Batch-encode audio into discrete codes (and optional conditioning, depending on 25Hz/12Hz).

        Args:
            audios (AudioInput):
                Supported forms:
                - np.ndarray: waveform (requires sr)
                - list[np.ndarray]: waveforms (requires sr)
                - str: wav path OR base64 audio string
                - list[str]: wav paths and/or base64 strings
            sr (Optional[int], default=None):
                Original sampling rate for numpy waveform input.
            return_dict (bool, default=True):
                Forwarded to model.encode(...). If True, returns ModelOutput.

        Returns:
            25Hz:
                Qwen3TTSTokenizerV1EncoderOutput (if return_dict=True) with fields:
                  - audio_codes: List[torch.LongTensor] each (codes_len,)
                  - xvectors:   List[torch.FloatTensor] each (xvector_dim,)
                  - ref_mels:   List[torch.FloatTensor] each (mel_len, mel_dim)
            12Hz:
                Qwen3TTSTokenizerV2EncoderOutput (if return_dict=True) with fields:
                  - audio_codes: List[torch.LongTensor] each (codes_len, num_quantizers)

            If return_dict=False, returns the raw tuple from model.encode.
        """
        model, feature_extractor, device = self._require_initialized()
        wavs = self._normalize_audio_inputs(audios, sr=sr)

        inputs_raw: object = feature_extractor(
            raw_audio=wavs,
            sampling_rate=int(feature_extractor.sampling_rate),
            return_tensors="pt",
        )
        if not isinstance(inputs_raw, BatchFeature):
            raise TypeError("Feature extractor output must be a BatchFeature.")
        inputs_converted: object = inputs_raw.to(device).to(model.dtype)
        if not isinstance(inputs_converted, BatchFeature):
            raise TypeError(
                "Converted feature extractor output must be a BatchFeature."
            )

        input_values_raw: object = inputs_converted.get("input_values")
        padding_mask_raw: object = inputs_converted.get("padding_mask")
        if not isinstance(input_values_raw, torch.Tensor):
            raise TypeError("BatchFeature `input_values` must be a tensor.")
        if not isinstance(padding_mask_raw, torch.Tensor):
            raise TypeError("BatchFeature `padding_mask` must be a tensor.")
        if input_values_raw.dim() not in (2, 3):
            raise ValueError("BatchFeature `input_values` must be 2-D or 3-D.")
        if padding_mask_raw.dim() not in (2, 3):
            raise ValueError("BatchFeature `padding_mask` must be 2-D or 3-D.")
        input_values = input_values_raw.squeeze(1)
        padding_mask = padding_mask_raw.squeeze(1)
        if input_values.dim() != 2 or padding_mask.dim() != 2:
            raise ValueError("BatchFeature audio tensors must have one channel.")
        if input_values.shape != padding_mask.shape:
            raise ValueError(
                "BatchFeature `input_values` and `padding_mask` shapes must match."
            )

        with torch.inference_mode():
            # model.encode expects (B, T) and (B, T)
            enc = model.encode(
                input_values,
                padding_mask,
                return_dict=return_dict,
            )
        return enc
