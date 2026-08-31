"""PyTorch Qwen3TTSTokenizerV1 model."""

import os
from dataclasses import dataclass, field
from typing import Literal, Protocol, SupportsIndex, cast, overload

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, logging
from transformers.utils.hub import cached_file
from typing_extensions import Self, override

from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV1DecoderConfig,
    Qwen3TTSTokenizerV1EncoderConfig,
)
from .modeling_qwen3_tts_tokenizer_v1_core import (
    Qwen3TTSTokenizerV1DecoderBigVGANModel,
    Qwen3TTSTokenizerV1DecoderDiTModel,
    Qwen3TTSTokenizerV1DecoderPreTrainedModel,
    Qwen3TTSTokenizerV1EncoderPreTrainedModel,
)
from .vq.speech_vq import WhisperEncoderVQ, XVectorExtractor
from .vq.whisper_encoder import get_mel_audio, get_T_after_cnn

logger = logging.get_logger(__name__)


class _PretrainedModelLoader(Protocol):
    def __call__(
        self,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: object,
        config: Qwen3TTSTokenizerV1Config | str | os.PathLike[str] | None,
        cache_dir: str | os.PathLike[str] | None,
        ignore_mismatched_sizes: bool,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        revision: str,
        use_safetensors: bool | None,
        weights_only: bool,
        proxies: dict[str, str] | None,
        **kwargs: object,
    ) -> object: ...


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i,).
    xvectors (`List[torch.FloatTensor]`):
        X-vector embeddings computed using `model.encode`, each tensor has shape (xvector_dim,).
    ref_mels (`List[torch.FloatTensor]`):
        Reference mel spectrogram computed using `model.encode`, each tensor has shape (mel_length_i, mel_dim,).
    """

    audio_codes: list[torch.Tensor] = field(default_factory=list)
    xvectors: list[torch.Tensor] = field(default_factory=list)
    ref_mels: list[torch.Tensor] = field(default_factory=list)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1DecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: list[torch.Tensor] = field(default_factory=list)


@auto_docstring
class Qwen3TTSTokenizerV1Decoder(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    _no_split_modules: list[str] = [  # noqa: RUF012
        "Qwen3TTSTokenizerV1DecoderDiTModel",
        "Qwen3TTSTokenizerV1DecoderBigVGANModel",
    ]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig) -> None:
        super().__init__(config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen3TTSTokenizerV1Decoder will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.dit = Qwen3TTSTokenizerV1DecoderDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.bigvgan = Qwen3TTSTokenizerV1DecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    @override
    def forward(
        self,
        code: torch.Tensor,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float | None = -1.0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Generates a waveform from input code and conditioning parameters."""

        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = cast(torch.Tensor, self.bigvgan(mel_spectrogram))

        return waveform


class Qwen3TTSTokenizerV1Encoder(Qwen3TTSTokenizerV1EncoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1EncoderConfig

    def __init__(self, config: Qwen3TTSTokenizerV1EncoderConfig) -> None:
        super().__init__(config)

        self.tokenizer = WhisperEncoderVQ(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_window=config.n_window,
            output_dim=config.output_dim,
            grad_checkpointing=config.grad_checkpointing,
            enable_mp=config.enable_mp,
            audio_sequence_parallel=config.audio_sequence_parallel,
            audio_vq_type=config.audio_vq_type,
            audio_vq_layers=config.audio_vq_layers,
            audio_vq_codebook_size=config.audio_vq_codebook_size,
            audio_vq_codebook_dim=config.audio_vq_codebook_dim,
            audio_vq_pe=config.audio_vq_pe,
            audio_vq_ds_rate=config.audio_vq_ds_rate,
        )

        self.padding = True
        self.audio_vq_ds_rate = self.tokenizer.audio_vq_ds_rate

    def speech2mel(self, speechs: list[torch.Tensor]) -> list[torch.Tensor]:
        mels = [
            get_mel_audio(
                speech, padding=self.padding, audio_vq_ds_rate=self.audio_vq_ds_rate
            )
            .to(speech.dtype)
            .to(self.tokenizer.conv1.weight.device)
            for speech in speechs
        ]
        return mels

    def mel2code(self, mels: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        audio_mellens = [mel.size(-1) for mel in mels]
        audio_aftercnnlens = [get_T_after_cnn(T) for T in audio_mellens]
        audio_seqlens = [T + 2 for T in audio_aftercnnlens]

        with torch.no_grad():
            _, indices = self.tokenizer(
                x_list=mels,
                audio_mellens=audio_mellens,
                audio_aftercnnlens=audio_aftercnnlens,
                audio_seqlens=audio_seqlens,
                return_indices=True,
            )

        indice_lens = [T // self.tokenizer.audio_vq_ds_rate for T in audio_aftercnnlens]
        indices = pad_sequence(
            list(torch.split(indices, indice_lens)), batch_first=True, padding_value=0
        )

        return indices, indice_lens

    def quantize_speech(
        self, speechs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]:
        mels = self.speech2mel(speechs)
        indices, indice_lens = self.mel2code(mels)
        return indices, indice_lens


@auto_docstring
class Qwen3TTSTokenizerV1PreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV1 model.
    """
)
class Qwen3TTSTokenizerV1Model(Qwen3TTSTokenizerV1PreTrainedModel):
    encoder_xvector_extractor: XVectorExtractor | None

    def __init__(self, config: Qwen3TTSTokenizerV1Config) -> None:
        super().__init__(config)
        self.config = config

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerV1Encoder._from_config(
            self.config.encoder_config
        )
        self.decoder = Qwen3TTSTokenizerV1Decoder._from_config(
            self.config.decoder_config
        )

        self.encoder_xvector_extractor = None

        self.post_init()

    def load_encoder_xvector_extractor(
        self, model_path: str | os.PathLike[str]
    ) -> None:
        self.encoder_xvector_extractor = XVectorExtractor(model_path)

    def get_model_type(self) -> str:
        return self.config.model_type

    def get_input_sample_rate(self) -> int:
        return self.input_sample_rate

    def get_output_sample_rate(self) -> int:
        return self.output_sample_rate

    def get_encode_downsample_rate(self) -> int:
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self) -> int:
        return self.decode_upsample_rate

    @classmethod
    @overload
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: object,
        config: Qwen3TTSTokenizerV1Config | str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        proxies: dict[str, str] | None = None,
        output_loading_info: Literal[False] = False,
        **kwargs: object,
    ) -> Self: ...

    @classmethod
    @overload
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: object,
        config: Qwen3TTSTokenizerV1Config | str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        proxies: dict[str, str] | None = None,
        output_loading_info: Literal[True] = True,
        **kwargs: object,
    ) -> tuple[Self, dict[str, object]]: ...

    @classmethod
    @override
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: object,
        config: Qwen3TTSTokenizerV1Config | str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        proxies: dict[str, str] | None = None,
        output_loading_info: bool = False,
        **kwargs: object,
    ) -> Self | tuple[Self, dict[str, object]]:
        loader = cast(_PretrainedModelLoader, super().from_pretrained)
        loaded = loader(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            proxies=proxies,
            output_loading_info=output_loading_info,
            **kwargs,
        )
        loading_info: dict[str, object] | None = None
        model_raw = loaded
        if output_loading_info:
            if (
                not isinstance(loaded, tuple)
                or len(loaded) != 2
                or not isinstance(loaded[1], dict)
            ):
                raise TypeError("Transformers returned invalid model loading info.")
            model_raw, loading_info_raw = loaded
            if not all(isinstance(key, str) for key in loading_info_raw):
                raise TypeError(
                    "Transformers returned loading info with non-string keys."
                )
            loading_info = cast(dict[str, object], loading_info_raw)
        if not isinstance(model_raw, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model_raw).__name__}.")
        model = model_raw
        subfolder = kwargs.get("subfolder")
        if subfolder is not None and not isinstance(subfolder, str):
            raise TypeError("`subfolder` must be a string.")
        encoder_xvector_extractor_path = cached_file(
            pretrained_model_name_or_path,
            "campplus.onnx",
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if encoder_xvector_extractor_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{encoder_xvector_extractor_path} not exists"""
            )
        model.load_encoder_xvector_extractor(encoder_xvector_extractor_path)

        if loading_info is not None:
            return model, loading_info
        return model

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | Qwen3TTSTokenizerV1EncoderOutput
    ):
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        if padding_mask is None:
            raise ValueError("`padding_mask` is required for encode.")

        wavs = [
            value[: int(mask.sum().item())]
            for value, mask in zip(input_values, padding_mask)
        ]

        code_batch, codes_lens = self.encoder.quantize_speech(wavs)
        code_rows = cast(tuple[torch.Tensor, ...], code_batch.unbind(0))
        codes = [code[: int(length)] for code, length in zip(code_rows, codes_lens)]

        xvectors: list[torch.Tensor] = []
        ref_mels: list[torch.Tensor] = []
        xvector_extractor = self.encoder_xvector_extractor
        if xvector_extractor is None:
            raise RuntimeError("Encoder xvector extractor is not initialized.")
        for wav in wavs:
            xvector, ref_mel = xvector_extractor.extract_code(wav.cpu().numpy())
            xvector = torch.tensor(xvector).to(wav.dtype).to(wav.device)
            ref_mel = torch.tensor(ref_mel).to(wav.dtype).to(wav.device)
            xvectors.append(xvector)
            ref_mels.append(ref_mel)

        if not return_dict:
            return (codes, xvectors, ref_mels)

        return Qwen3TTSTokenizerV1EncoderOutput(codes, xvectors, ref_mels)

    def decode(
        self,
        audio_codes: torch.Tensor,
        xvectors: torch.Tensor,
        ref_mels: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[list[torch.Tensor]] | Qwen3TTSTokenizerV1DecoderOutput:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            xvectors (`torch.FloatTensor` of shape `(batch_size, xvector_dim)`, *optional*):
                X-vector embeddings computed using `model.encode`.
            ref_mels (`torch.FloatTensor` of shape `(batch_size, mel_length, mel_dim)`, *optional*):
                Reference mel spectrogram computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        audio_lengths = (audio_codes > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)
        audio_batch = cast(
            torch.Tensor,
            self.decoder(
                code=audio_codes, reference_mel=ref_mels, conditioning=xvectors
            ),
        )

        audio_values = [
            audio[: cast(SupportsIndex, length)]
            for audio, length in zip(audio_batch.unbind(0), audio_lengths.unbind(0))
        ]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV1DecoderOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerV1Model", "Qwen3TTSTokenizerV1PreTrainedModel"]
