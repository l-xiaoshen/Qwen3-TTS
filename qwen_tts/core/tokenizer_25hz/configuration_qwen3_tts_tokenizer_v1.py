"""Qwen3TTSTokenizerV1 model configuration"""

from typing import ClassVar, Protocol, TypeVar, cast

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


_ConfigT = TypeVar("_ConfigT", bound=PretrainedConfig)
_ConfigT_co = TypeVar("_ConfigT_co", bound=PretrainedConfig, covariant=True)


class _ConfigFactory(Protocol[_ConfigT_co]):
    def __call__(self, **kwargs: object) -> _ConfigT_co: ...


class _ConfigInitializer(Protocol):
    def __call__(self, **kwargs: object) -> None: ...


def _build_config(config_type: type[_ConfigT], values: dict[str, object]) -> _ConfigT:
    return cast(_ConfigFactory[_ConfigT], config_type)(**values)


class Qwen3TTSTokenizerV1DecoderDiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Qwen3TTSTokenizerV1DecoderToken2WavDiT.
    It defines the architecture of the DiT model, which is used for generating mel-spectrograms from tokens.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            The dimension of the model.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            The number of transformer blocks in the DiT model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of attention heads in each transformer block.
        ff_mult (`int`, *optional*, defaults to 2):
            The multiplier for the feedforward layer in each transformer block.
        emb_dim (`int`, *optional*, defaults to 512):
            The dimension of the embedding layer.
        head_dim (`int`, *optional*, defaults to 64):
            The dimension of each attention head.
        repeats (`int`, *optional*, defaults to 2):
            The number of times the codec embeddings are repeated.
        num_embeds (`int`, *optional*, defaults to 8193):
            The number of unique embeddings in the codec.
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the transformer blocks.

        enc_emb_dim (`int`, *optional*, defaults to 192):
            The dimension of the pre-trained speaker embedding.
        enc_dim (`int`, *optional*, defaults to 128):
            The dimension of the encoder output.
        enc_channels (`list[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder.
        enc_attention_channels (`int`, *optional*, defaults to 64):
            The number of attention channels in the SqueezeExcitationBlock.
        enc_res2net_scale (`int`, *optional*, defaults to 2):
            The scale of the Res2Net block in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 64):
            The number of output channels after squeeze in the SqueezeExcitationBlock.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder_dit"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        ff_mult: int = 2,
        emb_dim: int = 512,
        head_dim: int = 64,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        block_size: int = 24,
        look_ahead_layers: list[int] | None = None,
        look_backward_layers: list[int] | None = None,
        repeats: int = 2,
        num_embeds: int = 8193,
        mel_dim: int = 80,
        dropout: float = 0.1,
        enc_emb_dim: int = 192,
        enc_dim: int = 128,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int = 64,
        enc_res2net_scale: int = 2,
        enc_se_channels: int = 64,
        **kwargs: object,
    ) -> None:
        if look_ahead_layers is None:
            look_ahead_layers = [10]
        if look_backward_layers is None:
            look_backward_layers = [0, 20]
        if enc_channels is None:
            enc_channels = [256, 256, 256, 256, 768]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 4, 1]

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.block_size = block_size
        self.look_ahead_layers = look_ahead_layers
        self.look_backward_layers = look_backward_layers
        self.repeats = repeats
        self.num_embeds = num_embeds
        self.mel_dim = mel_dim
        self.dropout = dropout
        self.enc_emb_dim = enc_emb_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_dilations = enc_dilations
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        cast(_ConfigInitializer, super().__init__)(**kwargs)


class Qwen3TTSTokenizerV1DecoderBigVGANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Qwen3TTSTokenizerV1DecoderToken2WavBigVGAN module.
    It defines the architecture of the BigVGAN model, which is used for converting mel-spectrograms to waveforms.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            The number of channels in the initial upsampling layer.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            A list of kernel sizes for each residual block.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A list of dilation sizes for each residual block.
        upsample_rates (`list[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            A list of upsampling rates for each upsampling layer.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            A list of kernel sizes for each upsampling layer.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder_bigvgan"

    def __init__(
        self,
        mel_dim: int = 80,
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        **kwargs: object,
    ) -> None:
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if upsample_rates is None:
            upsample_rates = [5, 3, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 7, 4, 4, 4, 4]

        self.mel_dim = mel_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        cast(_ConfigInitializer, super().__init__)(**kwargs)


class Qwen3TTSTokenizerV1DecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTokenizerV1DecoderConfig`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        dit_config ([`DiT_Args`], *optional*):
            Configuration class for the Diffusion Transformer (DiT) module responsible for generating mel-spectrograms.
        bigvgan_config ([`BigVGAN_Args`], *optional*):
            Configuration class for the BigVGAN module responsible for converting mel-spectrograms to waveforms.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "dit_config": Qwen3TTSTokenizerV1DecoderDiTConfig,
        "bigvgan_config": Qwen3TTSTokenizerV1DecoderBigVGANConfig,
    }

    def __init__(
        self,
        dit_config: dict[str, object] | None = None,
        bigvgan_config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        if dit_config is None:
            dit_config = {}
        if bigvgan_config is None:
            bigvgan_config = {}
        self.dit_config = _build_config(Qwen3TTSTokenizerV1DecoderDiTConfig, dit_config)
        self.bigvgan_config = _build_config(
            Qwen3TTSTokenizerV1DecoderBigVGANConfig, bigvgan_config
        )
        cast(_ConfigInitializer, super().__init__)(**kwargs)


class Qwen3TTSTokenizerV1EncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Qwen3TTSTokenizerV1 Encoder.

    The encoder typically takes mel-spectrogram features and produces high-level audio representations, then (optionally)
    applies an Audio-VQ module (e.g., GRVQ) to discretize continuous representations into codes.

    Args:
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel bins in the input mel-spectrogram.
        n_ctx (`int`, *optional*, defaults to 1500):
            Maximum input sequence length (in frames/tokens) for the encoder.
        n_state (`int`, *optional*, defaults to 1280):
            Hidden size (model dimension) of the encoder transformer.
        n_head (`int`, *optional*, defaults to 20):
            Number of attention heads in each transformer layer.
        n_layer (`int`, *optional*, defaults to 32):
            Number of transformer layers.
        n_window (`int`, *optional*, defaults to 100):
            Window size used by the model for local attention / chunking (implementation-dependent).
        output_dim (`int`, *optional*, defaults to 3584):
            Output feature dimension produced by the encoder head (before/after projection, implementation-dependent).

        grad_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing to reduce memory usage during training.
        enable_mp (`bool`, *optional*, defaults to `False`):
            Whether to enable model parallel features (implementation-dependent).
        audio_sequence_parallel (`bool`, *optional*, defaults to `False`):
            Whether to enable sequence parallelism for audio branch (implementation-dependent).

        audio_vq_type (`str`, *optional*, defaults to `"GRVQ"`):
            Type of audio vector-quantization module. Common choices: `"GRVQ"`, `"RVQ"`, etc.
        audio_vq_layers (`int`, *optional*, defaults to 6):
            Number of VQ layers / quantizers (e.g., number of residual quantizers for RVQ/GRVQ-like designs).
        audio_vq_codebook_size (`int`, *optional*, defaults to 32768):
            Size of each codebook (number of entries).
        audio_vq_codebook_dim (`int`, *optional*, defaults to 1280):
            Dimension of codebook vectors (often equals encoder hidden size).
        audio_vq_pe (`bool`, *optional*, defaults to `True`):
            Whether to use positional encoding (or position embeddings) inside the VQ module.
        audio_vq_ds_rate (`int`, *optional*, defaults to 2):
            Downsampling rate applied before VQ (e.g., temporal downsample factor).
    """

    model_type = "qwen3_tts_tokenizer_v1_encoder"

    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 1500,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        n_window: int = 100,
        output_dim: int = 3584,
        grad_checkpointing: bool = False,
        enable_mp: bool = False,
        audio_sequence_parallel: bool = False,
        audio_vq_type: str = "GRVQ",
        audio_vq_layers: int = 6,
        audio_vq_codebook_size: int = 32768,
        audio_vq_codebook_dim: int | None = 1280,
        audio_vq_pe: bool = True,
        audio_vq_ds_rate: int = 2,
        **kwargs: object,
    ) -> None:
        cast(_ConfigInitializer, super().__init__)(**kwargs)
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_window = n_window
        self.output_dim = output_dim
        self.grad_checkpointing = grad_checkpointing
        self.enable_mp = enable_mp
        self.audio_sequence_parallel = audio_sequence_parallel
        self.audio_vq_type = audio_vq_type
        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_codebook_dim = audio_vq_codebook_dim
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_ds_rate = audio_vq_ds_rate


class Qwen3TTSTokenizerV1Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3TTSTokenizerV1Config`]. It is used to instantiate a Qwen3TTSTokenizerV1Model
    model according to the specified sub-models configurations, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`dict`, *optional*): Configuration of the underlying encoder sub-model.
        decoder_config (`dict`, *optional*): Configuration of the underlying decoder sub-model.
    """

    model_type = "qwen3_tts_tokenizer_25hz"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "encoder_config": Qwen3TTSTokenizerV1EncoderConfig,
        "decoder_config": Qwen3TTSTokenizerV1DecoderConfig,
    }

    def __init__(
        self,
        encoder_config: dict[str, object] | None = None,
        decoder_config: dict[str, object] | None = None,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        decode_upsample_rate: int = 1920,
        encode_downsample_rate: int = 1920,
        **kwargs: object,
    ) -> None:
        cast(_ConfigInitializer, super().__init__)(**kwargs)
        if encoder_config is None:
            encoder_config = {}
            logger.info(
                "encoder_config is None. Initializing encoder with default values"
            )
        if decoder_config is None:
            decoder_config = {}
            logger.info(
                "decoder_config is None. Initializing decoder with default values"
            )

        self.encoder_config = _build_config(
            Qwen3TTSTokenizerV1EncoderConfig, encoder_config
        )
        self.decoder_config = _build_config(
            Qwen3TTSTokenizerV1DecoderConfig, decoder_config
        )

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate


__all__ = [
    "Qwen3TTSTokenizerV1Config",
    "Qwen3TTSTokenizerV1DecoderBigVGANConfig",
    "Qwen3TTSTokenizerV1DecoderConfig",
    "Qwen3TTSTokenizerV1DecoderDiTConfig",
    "Qwen3TTSTokenizerV1EncoderConfig",
]
