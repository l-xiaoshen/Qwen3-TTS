import typing as tp
from collections.abc import Callable, Mapping

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from typing_extensions import override


class _TokenizerProtocol(tp.Protocol):
    init_kwargs: Mapping[str, object]
    model_input_names: list[str]

    def __call__(
        self, text: list[str] | list[list[str]], **kwargs: object
    ) -> Mapping[str, object]: ...

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> list[str]: ...

    def decode(
        self,
        token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> str | list[str]: ...


class Qwen3TTSProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3TTS processor.

    Args:
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes: tp.ClassVar[list[str]] = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    tokenizer: _TokenizerProtocol

    def __init__(
        self, tokenizer: _TokenizerProtocol, chat_template: str | None = None
    ) -> None:
        super().__init__(tokenizer, chat_template=chat_template)

    @override
    def __call__(
        self, text: str | list[str] | list[list[str]], **kwargs: object
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """

        text_inputs = text if isinstance(text, list) else [text]
        merge_kwargs = tp.cast(
            Callable[..., dict[str, dict[str, object]]], self._merge_kwargs
        )
        output_kwargs = merge_kwargs(
            ProcessingKwargs,
            tokenizer_init_kwargs=dict(self.tokenizer.init_kwargs),
            **kwargs,
        )

        text_kwargs: dict[str, object] = {"padding": False, "padding_side": "left"}
        text_kwargs.update(output_kwargs.get("text_kwargs", {}))
        return_tensors = output_kwargs.get("common_kwargs", {}).get("return_tensors")
        texts_inputs = self.tokenizer(text_inputs, **text_kwargs)
        tensor_type = return_tensors if isinstance(return_tensors, str) else None

        return BatchFeature(
            data={**texts_inputs},
            tensor_type=tensor_type,
        )

    @override
    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> list[str]:
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @override
    def decode(
        self,
        token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> str | list[str]:
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    @override
    def model_input_names(self) -> list[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names))


__all__ = ["Qwen3TTSProcessor"]
