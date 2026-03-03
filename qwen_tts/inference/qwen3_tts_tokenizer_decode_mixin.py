# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Qwen3TTSTokenizerDecodeMixin:
    def decode(
        self,
        encoded,
    ) -> Tuple[List[np.ndarray], int]:
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
            encoded (Any):
                - ModelOutput returned by `encode()`, OR
                - dict, OR
                - list[dict]

        Returns:
            Tuple[List[np.ndarray], int]:
                - wavs: list of 1-D float32 numpy arrays
                - sample_rate: int, model output sampling rate
        """
        model_type = self.model.get_model_type()

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        # Normalize `encoded` into the same shapes as the official demo uses.
        if hasattr(encoded, "audio_codes"):
            # ModelOutput from encode()
            audio_codes_list = encoded.audio_codes
            xvectors_list = getattr(encoded, "xvectors", None)
            ref_mels_list = getattr(encoded, "ref_mels", None)
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
            xvectors_list = encoded.get("xvectors", None)
            ref_mels_list = encoded.get("ref_mels", None)
        elif isinstance(encoded, list):
            # list of dicts
            audio_codes_list = [e["audio_codes"] for e in encoded]
            xvectors_list = (
                [e["xvectors"] for e in encoded] if ("xvectors" in encoded[0]) else None
            )
            ref_mels_list = (
                [e["ref_mels"] for e in encoded] if ("ref_mels" in encoded[0]) else None
            )
        else:
            raise TypeError(
                "`encoded` must be an encode output, a dict, or a list of dicts."
            )

        # Ensure list form for per-sample tensors
        if isinstance(audio_codes_list, torch.Tensor):
            # Could be a single sample tensor or an already padded batch tensor.
            t = audio_codes_list
            if t.dim() == 1:
                # 25Hz single sample: (C,) -> (1, C)
                t = t.unsqueeze(0)
            elif t.dim() == 2:
                # 12Hz single sample: (C, Q) -> (1, C, Q)
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:
            # List[Tensor/np]
            audio_codes_list = [
                _to_tensor(c, dtype=torch.long) for c in audio_codes_list
            ]
            audio_codes_padded = pad_sequence(
                audio_codes_list, batch_first=True, padding_value=-1
            ).to(self.device)

        with torch.inference_mode():
            if model_type == "qwen3_tts_tokenizer_25hz":
                if xvectors_list is None or ref_mels_list is None:
                    raise ValueError("25Hz decode requires `xvectors` and `ref_mels`.")

                if isinstance(xvectors_list, torch.Tensor):
                    xvectors_batch = xvectors_list
                    if xvectors_batch.dim() == 1:  # (D,) -> (1, D)
                        xvectors_batch = xvectors_batch.unsqueeze(0)
                    xvectors_batch = xvectors_batch.to(self.device).to(self.model.dtype)
                else:
                    xvectors_list = [
                        _to_tensor(x, dtype=torch.float32) for x in xvectors_list
                    ]
                    xvectors_batch = (
                        torch.stack(xvectors_list, dim=0)
                        .to(self.device)
                        .to(self.model.dtype)
                    )

                if isinstance(ref_mels_list, torch.Tensor):
                    ref_mels_padded = ref_mels_list
                    if ref_mels_padded.dim() == 2:  # (T, M) -> (1, T, M)
                        ref_mels_padded = ref_mels_padded.unsqueeze(0)
                    ref_mels_padded = ref_mels_padded.to(self.device).to(
                        self.model.dtype
                    )
                else:
                    ref_mels_list = [
                        _to_tensor(m, dtype=torch.float32) for m in ref_mels_list
                    ]
                    ref_mels_padded = (
                        pad_sequence(ref_mels_list, batch_first=True, padding_value=0)
                        .to(self.device)
                        .to(self.model.dtype)
                    )

                dec = self.model.decode(
                    audio_codes_padded,
                    xvectors_batch,
                    ref_mels_padded,
                    return_dict=True,
                )
                wav_tensors = dec.audio_values

            elif model_type == "qwen3_tts_tokenizer_12hz":
                dec = self.model.decode(audio_codes_padded, return_dict=True)
                wav_tensors = dec.audio_values

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(self.model.get_output_sample_rate())

    def get_model_type(self) -> str:
        """
        Get the underlying tokenizer model type.

        Returns:
            str: Model type string from `self.model.config.model_type`
                (e.g. "qwen3_tts_tokenizer_25hz" / "qwen3_tts_tokenizer_12hz").
        """
        return self.model.get_model_type()

    def get_input_sample_rate(self) -> int:
        """
        Get the expected input sample rate for encoding.

        Returns:
            int: Input sample rate (Hz).
        """
        return int(self.model.get_input_sample_rate())

    def get_output_sample_rate(self) -> int:
        """
        Get the output sample rate for decoded waveforms.

        Returns:
            int: Output sample rate (Hz).
        """
        return int(self.model.get_output_sample_rate())

    def get_encode_downsample_rate(self) -> int:
        """
        Get the encoder downsample rate (waveform samples per code step).

        Returns:
            int: Encode downsample rate.
        """
        return int(self.model.get_encode_downsample_rate())

    def get_decode_upsample_rate(self) -> int:
        """
        Get the decoder upsample rate (waveform samples per code step).

        Returns:
            int: Decode upsample rate.
        """
        return int(self.model.get_decode_upsample_rate())
