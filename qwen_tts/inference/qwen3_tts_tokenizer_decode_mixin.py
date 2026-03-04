# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


@runtime_checkable
class _TokenizerDecodeOutput(Protocol):
    audio_values: torch.Tensor | Sequence[torch.Tensor]


@runtime_checkable
class _TokenizerDecodeModel(Protocol):
    dtype: torch.dtype

    def get_model_type(self) -> str: ...
    def get_input_sample_rate(self) -> int: ...
    def get_output_sample_rate(self) -> int: ...
    def get_encode_downsample_rate(self) -> int: ...
    def get_decode_upsample_rate(self) -> int: ...
    def decode(self, *args: object, **kwargs: object) -> _TokenizerDecodeOutput: ...


def _to_tensor(value: object, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        raise TypeError(f"Expected tensor-like value, got {type(value)}")
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _prepare_audio_codes_batch(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        audio_codes = _to_tensor(value, dtype=torch.long)
        if audio_codes.dim() in (1, 2):
            audio_codes = audio_codes.unsqueeze(0)
        return audio_codes.to(device)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        audio_code_list: list[torch.Tensor] = []
        for item in value:
            audio_code_list.append(_to_tensor(item, dtype=torch.long))
        if len(audio_code_list) == 0:
            raise ValueError("`audio_codes` list is empty.")
        return pad_sequence(audio_code_list, batch_first=True, padding_value=-1).to(
            device
        )

    raise TypeError(
        "`audio_codes` must be a tensor/ndarray or a sequence of tensor-like values."
    )


def _prepare_xvectors_batch(
    value: object, device: torch.device, model_dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        xvectors = _to_tensor(value, dtype=torch.float32)
        if xvectors.dim() == 1:
            xvectors = xvectors.unsqueeze(0)
        return xvectors.to(device).to(model_dtype)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        xvector_items: list[torch.Tensor] = []
        for item in value:
            xvec = _to_tensor(item, dtype=torch.float32)
            if xvec.dim() == 2 and xvec.shape[0] == 1:
                xvec = xvec.squeeze(0)
            if xvec.dim() != 1:
                raise ValueError("Each xvector item must be 1-D or shaped as (1, D).")
            xvector_items.append(xvec)
        if len(xvector_items) == 0:
            raise ValueError("`xvectors` list is empty.")
        return torch.stack(xvector_items, dim=0).to(device).to(model_dtype)

    raise TypeError(
        "`xvectors` must be a tensor/ndarray or a sequence of tensor-like values."
    )


def _prepare_ref_mels_batch(
    value: object, device: torch.device, model_dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        ref_mels = _to_tensor(value, dtype=torch.float32)
        if ref_mels.dim() == 2:
            ref_mels = ref_mels.unsqueeze(0)
        return ref_mels.to(device).to(model_dtype)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        ref_mel_items: list[torch.Tensor] = []
        for item in value:
            ref_mel = _to_tensor(item, dtype=torch.float32)
            if ref_mel.dim() == 3 and ref_mel.shape[0] == 1:
                ref_mel = ref_mel.squeeze(0)
            if ref_mel.dim() != 2:
                raise ValueError(
                    "Each ref_mels item must be 2-D or shaped as (1, T, M)."
                )
            ref_mel_items.append(ref_mel)
        if len(ref_mel_items) == 0:
            raise ValueError("`ref_mels` list is empty.")
        return (
            pad_sequence(ref_mel_items, batch_first=True, padding_value=0)
            .to(device)
            .to(model_dtype)
        )

    raise TypeError(
        "`ref_mels` must be a tensor/ndarray or a sequence of tensor-like values."
    )


def _extract_encoded_fields(
    encoded: object,
) -> tuple[object, object | None, object | None]:
    if hasattr(encoded, "audio_codes"):
        audio_codes = getattr(encoded, "audio_codes")
        xvectors = getattr(encoded, "xvectors", None)
        ref_mels = getattr(encoded, "ref_mels", None)
        return audio_codes, xvectors, ref_mels

    if isinstance(encoded, dict):
        encoded_map: dict[str, object] = {}
        for key, value in encoded.items():
            encoded_map[str(key)] = value
        if "audio_codes" not in encoded_map:
            raise KeyError("`encoded` mapping must contain key `audio_codes`.")
        audio_codes = encoded_map["audio_codes"]
        xvectors = encoded_map.get("xvectors")
        ref_mels = encoded_map.get("ref_mels")
        return audio_codes, xvectors, ref_mels

    if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
        encoded_items = list(encoded)
        if len(encoded_items) == 0:
            raise ValueError("`encoded` list is empty.")
        mapping_items: list[dict[str, object]] = []
        for item in encoded_items:
            if not isinstance(item, dict):
                raise TypeError("`encoded` list elements must be mapping-like objects.")
            normalized_item: dict[str, object] = {}
            for key, value in item.items():
                normalized_item[str(key)] = value
            mapping_items.append(normalized_item)
        audio_codes = [item["audio_codes"] for item in mapping_items]
        xvectors = None
        if all("xvectors" in item for item in mapping_items):
            xvectors = [item["xvectors"] for item in mapping_items]
        ref_mels = None
        if all("ref_mels" in item for item in mapping_items):
            ref_mels = [item["ref_mels"] for item in mapping_items]
        return audio_codes, xvectors, ref_mels

    raise TypeError(
        "`encoded` must be an encode output, a mapping, or a list of mappings."
    )


def _audio_values_to_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.dim() == 1:
            return [value]
        return [value[i] for i in range(value.shape[0])]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        wav_tensors: list[torch.Tensor] = []
        for item in value:
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    f"Decoded audio contains unsupported item type: {type(item)}"
                )
            wav_tensors.append(item)
        return wav_tensors

    raise TypeError(f"Unsupported decoded audio container type: {type(value)}")


class Qwen3TTSTokenizerDecodeMixin:
    def decode(
        self,
        encoded: object,
    ) -> tuple[list[np.ndarray], int]:
        """
        Decode back to waveform.

        Usage:
        1) Pass the raw output of `encode(...)` directly (recommended).
           - 25Hz: expects fields audio_codes, xvectors, ref_mels
           - 12Hz: expects field audio_codes
        2) Pass a dict or list[dict] (minimal form) for custom pipelines:
           - 25Hz dict keys: {"audio_codes", "xvectors", "ref_mels"}
           - 12Hz dict keys: {"audio_codes"}
           Values can be torch tensors or numpy arrays.

        Args:
            encoded (object):
                - ModelOutput returned by `encode()`, OR
                - dict, OR
                - list[dict]

        Returns:
            Tuple[List[np.ndarray], int]:
                - wavs: list of 1-D float32 numpy arrays
                - sample_rate: int, model output sampling rate
        """
        model = getattr(self, "model", None)
        device = getattr(self, "device", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        if not isinstance(device, torch.device):
            raise RuntimeError("Tokenizer device is not initialized.")
        model_type = model.get_model_type()
        audio_codes_raw, xvectors_raw, ref_mels_raw = _extract_encoded_fields(encoded)
        audio_codes_padded = _prepare_audio_codes_batch(audio_codes_raw, device)

        with torch.inference_mode():
            if model_type == "qwen3_tts_tokenizer_25hz":
                if xvectors_raw is None or ref_mels_raw is None:
                    raise ValueError("25Hz decode requires `xvectors` and `ref_mels`.")
                xvectors_batch = _prepare_xvectors_batch(
                    xvectors_raw, device, model.dtype
                )
                ref_mels_padded = _prepare_ref_mels_batch(
                    ref_mels_raw, device, model.dtype
                )

                dec = model.decode(
                    audio_codes_padded,
                    xvectors_batch,
                    ref_mels_padded,
                    return_dict=True,
                )
            elif model_type == "qwen3_tts_tokenizer_12hz":
                dec = model.decode(audio_codes_padded, return_dict=True)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        wav_tensors = _audio_values_to_tensors(dec.audio_values)
        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(model.get_output_sample_rate())

    def get_model_type(self) -> str:
        """
        Get the underlying tokenizer model type.

        Returns:
            str: Model type string from `self.model.config.model_type`
                (e.g. "qwen3_tts_tokenizer_25hz" / "qwen3_tts_tokenizer_12hz").
        """
        model = getattr(self, "model", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        return model.get_model_type()

    def get_input_sample_rate(self) -> int:
        """
        Get the expected input sample rate for encoding.

        Returns:
            int: Input sample rate (Hz).
        """
        model = getattr(self, "model", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        return int(model.get_input_sample_rate())

    def get_output_sample_rate(self) -> int:
        """
        Get the output sample rate for decoded waveforms.

        Returns:
            int: Output sample rate (Hz).
        """
        model = getattr(self, "model", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        return int(model.get_output_sample_rate())

    def get_encode_downsample_rate(self) -> int:
        """
        Get the encoder downsample rate (waveform samples per code step).

        Returns:
            int: Encode downsample rate.
        """
        model = getattr(self, "model", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        return int(model.get_encode_downsample_rate())

    def get_decode_upsample_rate(self) -> int:
        """
        Get the decoder upsample rate (waveform samples per code step).

        Returns:
            int: Decode upsample rate.
        """
        model = getattr(self, "model", None)
        if not isinstance(model, _TokenizerDecodeModel):
            raise RuntimeError("Tokenizer model is not initialized.")
        return int(model.get_decode_upsample_rate())
