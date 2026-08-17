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
import copy
import operator
import os
from collections.abc import Mapping, Sequence
from itertools import accumulate
from typing import Literal, Protocol, cast, overload

import numpy as np
import numpy.typing as npt
import onnxruntime
import sox
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor, nn
from torchaudio.compliance import kaldi
from typing_extensions import override

from .core_vq import DistributedGroupResidualVectorQuantization
from .whisper_encoder import (
    Conv1d,
    ConvTranspose1d,
    ResidualAttentionBlock,
    WhisperEncoder,
)

NumpyArray = npt.NDArray[np.generic]


class _OnnxInput(Protocol):
    name: str


class _OnnxGraphOptimizationLevel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class _OnnxGraphOptimizationLevels(Protocol):
    ORT_ENABLE_ALL: _OnnxGraphOptimizationLevel


class _OnnxSessionOptions(Protocol):
    graph_optimization_level: _OnnxGraphOptimizationLevel
    intra_op_num_threads: int


class _OnnxSessionOptionsFactory(Protocol):
    def __call__(self) -> _OnnxSessionOptions: ...


class _OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[_OnnxInput]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NumpyArray],
    ) -> Sequence[object]: ...


class _SoxTransformer(Protocol):
    def norm(self, db_level: float) -> None: ...

    def build_array(
        self, *, input_array: NumpyArray, sample_rate_in: int
    ) -> NumpyArray: ...


def dynamic_range_compression_torch(
    x: torch.Tensor, C: float = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


class MelSpectrogramFeatures(nn.Module):
    """
    Calculate the BigVGAN style mel spectrogram of an input signal.
    Args:
        filter_length (int): The number of samples in the filter window, used for the Fourier Transform. Default is 1024.
        hop_length (int): The number of samples between successive frames (stride of the STFT). Default is 160.
        win_length (int): The length of the window function applied to each frame, usually less than or equal to the filter length. Default is 640.
        n_mel_channels (int): The number of Mel-frequency channels to output from the Mel-scale spectrogram. Default is 80.
        mel_fmin (int): The minimum frequency (in Hz) of the Mel-scale spectrogram. Default is 0.
        mel_fmax (int): The maximum frequency (in Hz) of the Mel-scale spectrogram. Default is 8000.
        sampling_rate (int): The sampling rate of the audio data (in Hz). Default is 16000.
        sampling_rate_org (int, optional): The original sampling rate of the audio data before any resampling (in Hz), if applicable. Default is None.
        padding (str): The padding mode for the input signal. 'center' pads the signal symmetrically around its center. Default is 'center'.

    Returns:
        torch.Tensor: Mel spectrogram.
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 160,
        win_length: int = 640,
        n_mel_channels: int = 80,
        mel_fmin: float = 0,
        mel_fmax: float = 8000,
        sampling_rate: int = 16000,
        sampling_rate_org: int | None = None,
        padding: Literal["center", "same"] = "center",
        use_db: bool = False,
    ) -> None:
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.sampling_rate_org = (
            sampling_rate_org if sampling_rate_org is not None else sampling_rate
        )
        self.mel_basis: dict[str, torch.Tensor] = {}
        self.hann_window: dict[str, torch.Tensor] = {}

    @override
    def forward(self, audio: torch.Tensor, **kwargs: object) -> torch.Tensor:
        with torch.no_grad():
            feats = self.extract(audio, **kwargs)
        return feats

    def extract(self, audio: torch.Tensor, **kwargs: object) -> torch.Tensor:

        if len(audio.shape) == 3:
            audio = audio.squeeze(1) if audio.shape[1] == 1 else audio.squeeze(2)
        if audio.ndim != 2:
            raise ValueError("Mel extraction expects a two-dimensional audio batch.")

        y = audio
        if len(list(self.mel_basis.keys())) == 0:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel_channels,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], spec
        )
        spec = spectral_normalize_torch(spec)

        return spec


class XVectorExtractor(nn.Module):
    def __init__(self, audio_codec_with_xvector: str | os.PathLike[str]) -> None:
        super().__init__()
        session_options_factory = cast(
            _OnnxSessionOptionsFactory,
            vars(onnxruntime)["SessionOptions"],
        )
        graph_optimization_levels = cast(
            _OnnxGraphOptimizationLevels,
            vars(onnxruntime)["GraphOptimizationLevel"],
        )
        option = session_options_factory()
        option.graph_optimization_level = graph_optimization_levels.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.ort_session = cast(
            _OnnxSession,
            onnxruntime.InferenceSession(
                audio_codec_with_xvector, sess_options=option, providers=providers
            ),
        )

        self.tfm = cast(_SoxTransformer, sox.Transformer())
        self.tfm.norm(db_level=-6)

        self.mel_ext = MelSpectrogramFeatures(
            filter_length=1024,
            hop_length=160,
            win_length=640,
            n_mel_channels=80,
            mel_fmin=0,
            mel_fmax=8000,
            sampling_rate=16000,
        )

    def extract_code(self, audio: NumpyArray) -> tuple[NumpyArray, NumpyArray]:
        with torch.no_grad():
            norm_audio = self.sox_norm(audio)

            norm_audio = torch.from_numpy(copy.deepcopy(norm_audio)).unsqueeze(0)
            feat = kaldi.fbank(
                norm_audio, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            ort_outputs = self.ort_session.run(
                None,
                {
                    self.ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )
            if not ort_outputs:
                raise TypeError("ONNX x-vector inference returned an invalid result.")
            norm_embedding = np.asarray(ort_outputs[0])
            if not np.issubdtype(norm_embedding.dtype, np.number):
                raise TypeError("ONNX x-vector inference must return numeric values.")
            norm_embedding = norm_embedding.flatten()
            norm_embedding = F.normalize(torch.from_numpy(norm_embedding), dim=0)

            ref_mel = self.mel_ext.extract(audio=norm_audio)

        return norm_embedding.numpy(), ref_mel.permute(0, 2, 1).squeeze(0).numpy()

    def sox_norm(self, audio: NumpyArray) -> NumpyArray:
        wav_norm = self.tfm.build_array(input_array=audio, sample_rate_in=16000)
        return wav_norm


WhisperEncoderVQOutput = tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]]


class WhisperEncoderVQ(WhisperEncoder[WhisperEncoderVQOutput]):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        n_window: int = 1500,
        output_dim: int = 512,
        grad_checkpointing: bool = False,
        enable_mp: bool = False,
        audio_sequence_parallel: bool = False,
        audio_vq_layers: int = -1,
        audio_vq_type: str = "NULL",
        audio_vq_codebook_size: int = 4096,
        audio_vq_pe: bool = False,
        audio_vq_commit_loss: float = 0.0,
        audio_vq_out_commit_loss: float = 0.0,
        audio_vq_no_quantize: bool = False,
        audio_vq_ff_layer: int = 0,
        audio_vq_threshold_ema_dead_code: float = 0.1,
        audio_vq_codebook_dim: int | None = None,
        audio_vq_ds_rate: int | None = None,
    ) -> None:
        super().__init__(
            n_mels,
            n_ctx,
            n_state,
            n_head,
            n_layer,
            n_window,
            output_dim,
            grad_checkpointing,
            enable_mp,
            audio_sequence_parallel,
        )

        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_commit_loss = audio_vq_commit_loss
        self.audio_vq_out_commit_loss = audio_vq_out_commit_loss
        self.audio_vq_no_quantize = audio_vq_no_quantize
        self.audio_vq_ff_layer = audio_vq_ff_layer

        if 0 < audio_vq_layers <= n_layer:
            self.vq_feature_dim = self.n_state
            self.audio_vq_ds_rate = 1
        else:
            raise ValueError(f"Unsupported audio_vq_layers: {audio_vq_layers}")

        if audio_vq_ds_rate is None:
            raise ValueError("`audio_vq_ds_rate` must be provided.")
        if self.audio_vq_ds_rate == audio_vq_ds_rate:
            self.audio_vq_downsample = nn.Identity()
            self.audio_vq_upsample = nn.Identity()
        else:
            if audio_vq_ds_rate % self.audio_vq_ds_rate != 0:
                raise ValueError(
                    "`audio_vq_ds_rate` must be divisible by the encoder VQ rate."
                )
            stride = audio_vq_ds_rate // self.audio_vq_ds_rate
            self.audio_vq_downsample = Conv1d(
                self.vq_feature_dim,
                self.vq_feature_dim,
                kernel_size=stride,
                stride=stride,
            )
            self.audio_vq_upsample = ConvTranspose1d(
                self.vq_feature_dim,
                self.vq_feature_dim,
                kernel_size=stride,
                stride=stride,
            )
            self.audio_vq_ds_rate = audio_vq_ds_rate

        if audio_vq_type != "GRVQ":
            raise ValueError(f"Unsupported audio_vq_type: {audio_vq_type}")
        self.audio_vq_type: Literal["GRVQ"] = "GRVQ"
        self.audio_quantizer = DistributedGroupResidualVectorQuantization(
            codebook_size=audio_vq_codebook_size,
            dim=self.vq_feature_dim,
            codebook_dim=audio_vq_codebook_dim,
            num_groups=1,
            num_quantizers=1,
            kmeans_init=False,
            threshold_ema_dead_code=audio_vq_threshold_ema_dead_code,
        )

        self.project_after_vq_pe: nn.Linear | None = None
        if self.audio_vq_pe:
            self.project_after_vq_pe = nn.Linear(self.n_state, self.n_state)

    def _calc_quantize_activities(
        self, indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        indices_onehot = F.one_hot(
            indices.long().flatten(), self.audio_vq_codebook_size
        ).sum(dim=0)
        vq_num_activities = (indices_onehot > 0).sum()
        vq_num_tokens = indices_onehot.sum()
        return {
            "vq_num_activities": vq_num_activities,
            "vq_num_tokens": vq_num_tokens,
        }

    def _do_quantize(
        self,
        x: torch.Tensor,
        pe: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: torch.Tensor, shape = (T, D)
        q: torch.Tensor, shape = (T, D)
        i: torch.Tensor, shape = (T)
        """
        x_teacher = x.clone() if self.audio_vq_out_commit_loss > 0 else None
        x = x.unsqueeze(0)

        x = cast(torch.Tensor, self.audio_vq_downsample(x.transpose(1, 2)))
        x = x.transpose(1, 2)

        vq_stats: dict[str, torch.Tensor] = {}

        if self.training:
            raise RuntimeError(
                "Training mode quantization is not supported for this VQ path. "
                "Use eval mode for inference."
            )
        indices = self.audio_quantizer.encode(x)
        x = self.audio_quantizer.decode(indices)
        indices = indices.squeeze(2).squeeze(1)

        vq_stats.update(self._calc_quantize_activities(indices))

        x, indices = x.squeeze(0), indices.squeeze(0)
        if self.audio_vq_pe:
            if pe is None:
                raise ValueError("`pe` must be provided when `audio_vq_pe` is enabled.")
            x = x + pe
            project_after_vq_pe = cast(nn.Linear, self.project_after_vq_pe)
            x = cast(torch.Tensor, project_after_vq_pe(x))

        x = cast(torch.Tensor, self.audio_vq_upsample(x.unsqueeze(0).transpose(1, 2)))
        x = x.transpose(1, 2).squeeze(0)

        if x_teacher is not None:
            vq_out_commit_loss = F.mse_loss(x_teacher.detach(), x)
            vq_stats["vq_out_commit_loss"] = (
                vq_out_commit_loss * self.audio_vq_out_commit_loss
            )

        return x, indices, vq_stats

    @overload
    def forward(
        self,
        x_list: list[Tensor],
        audio_mellens: list[int],
        audio_aftercnnlens: list[int],
        audio_seqlens: list[int],
        return_indices: Literal[True],
        audio_pitchs: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x_list: list[Tensor],
        audio_mellens: list[int],
        audio_aftercnnlens: list[int],
        audio_seqlens: list[int],
        return_indices: Literal[False] = False,
        audio_pitchs: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

    @overload
    def forward(
        self,
        x_list: list[Tensor],
        audio_mellens: list[int],
        audio_aftercnnlens: list[int],
        audio_seqlens: list[int],
        return_indices: bool,
        audio_pitchs: list[Tensor] | None = None,
    ) -> WhisperEncoderVQOutput: ...

    @override
    def forward(
        self,
        x_list: list[Tensor],
        audio_mellens: list[int],
        audio_aftercnnlens: list[int],
        audio_seqlens: list[int],
        return_indices: bool = False,
        audio_pitchs: list[Tensor] | None = None,
    ) -> WhisperEncoderVQOutput:
        """
        x : torch.Tensor, shape = (n_mels, n_ctx)
            the mel spectrogram of the audio
        """

        positional_embedding = self.positional_embedding
        audio_vq_ds_rate = self.audio_vq_ds_rate
        aftercnn_x_list: list[torch.Tensor] = []
        pe_for_vq_list: list[torch.Tensor] = []
        for each_x in x_list:
            each_x_split_list = each_x.split(self.n_window * 2, dim=1)
            for each_x_split in each_x_split_list:
                each_x_split = F.gelu(self.conv1(each_x_split))
                each_x_split = F.gelu(self.conv2(each_x_split))
                each_x_split = each_x_split.permute(1, 0)  # L,D

                each_positional_embedding_split = positional_embedding[
                    : each_x_split.shape[0]
                ]
                aftercnn_x_list.append(
                    each_x_split
                    + each_positional_embedding_split.to(each_x_split.dtype)
                )

                pe_for_vq_split = positional_embedding[
                    : each_x_split.shape[0] // audio_vq_ds_rate
                ]
                pe_for_vq_list.append(pe_for_vq_split.to(each_x_split.dtype))

        pe_for_vq = torch.cat(pe_for_vq_list, dim=0)
        x = torch.cat(aftercnn_x_list, dim=0)

        output_list: list[int] = []
        for item in audio_aftercnnlens:
            while item > self.n_window:
                output_list.append(self.n_window)
                item -= self.n_window
            output_list.append(item)

        cu_seqlens = list(accumulate(output_list, func=operator.add, initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens).to(device=x.device, dtype=torch.int32)

        blocks = list(self.blocks)
        vq_layer_index = self.audio_vq_layers - 1
        for block_module in blocks[:vq_layer_index]:
            block = cast(ResidualAttentionBlock, block_module)
            x = block(x, cu_seqlens=cu_seqlens)

        vq_block = cast(ResidualAttentionBlock, blocks[vq_layer_index])
        x = vq_block(x, cu_seqlens=cu_seqlens)
        x, indices, vq_stats = self._do_quantize(x, pe_for_vq)
        if return_indices:
            return x, indices

        for block_module in blocks[vq_layer_index + 1 :]:
            block = cast(ResidualAttentionBlock, block_module)
            x = block(x, cu_seqlens=cu_seqlens)

        pooled_x_list = x.split(audio_aftercnnlens, dim=0)
        token_x_list: list[torch.Tensor] = []
        for pooled_x in pooled_x_list:
            pooled_x = pooled_x.permute(1, 0)
            pooled_x = self.avg_pooler(pooled_x)
            pooled_x = pooled_x.permute(1, 0)
            token_x_list.append(pooled_x)
        x = torch.cat(token_x_list, dim=0)

        x = self.ln_post(x)

        x = self.proj(x)

        output = torch.zeros(
            (x.size(0) + len(audio_seqlens) * 2, x.size(1)),
            device=x.device,
            dtype=x.dtype,
        )

        audio_seqlens_acc = list(
            accumulate(audio_seqlens, func=operator.add, initial=0)
        )
        start_ids = torch.tensor(
            audio_seqlens_acc[:-1], device=x.device, dtype=torch.int32
        )
        end_ids = (
            torch.tensor(audio_seqlens_acc[1:], device=x.device, dtype=torch.int32) - 1
        )

        audio_tokens_mask = torch.ones(
            output.size(0), device=x.device, dtype=torch.bool
        )
        audio_tokens_mask[start_ids] = False
        audio_tokens_mask[end_ids] = False
        output[start_ids] = self.audio_bos_eos_token.weight[0].to(x.dtype)
        output[end_ids] = self.audio_bos_eos_token.weight[1].to(x.dtype)
        output[audio_tokens_mask] = x

        return output, vq_stats
