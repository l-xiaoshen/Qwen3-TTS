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
from collections.abc import Mapping, Sequence
import typing as tp

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


@tp.runtime_checkable
class _TokenizerProtocol(tp.Protocol):
    init_kwargs: Mapping[str, object]
    model_input_names: list[str]

    def __call__(
        self, text: Sequence[object], **kwargs: object
    ) -> Mapping[str, object]: ...

    def batch_decode(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, *args: object, **kwargs: object) -> object: ...


class Qwen3TTSProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3TTS processor.

    Args:
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer=None, chat_template=None):
        super().__init__(tokenizer, chat_template=chat_template)

    def _require_tokenizer(self) -> _TokenizerProtocol:
        tokenizer = getattr(self, "tokenizer", None)
        if not isinstance(tokenizer, _TokenizerProtocol):
            raise RuntimeError("Tokenizer is not initialized.")
        return tokenizer

    def __call__(self, *args: object, **kwargs: object) -> BatchFeature:
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

        tokenizer = self._require_tokenizer()
        text = kwargs.pop("text", args[0] if args else None)
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        text_inputs = text if isinstance(text, list) else [text]

        merge_kwargs_fn = getattr(self, "_merge_kwargs", None)
        if not callable(merge_kwargs_fn):
            raise RuntimeError("Processor kwargs merge helper is not available.")
        output_kwargs_raw = merge_kwargs_fn(
            ProcessingKwargs,
            tokenizer_init_kwargs=dict(tokenizer.init_kwargs),
            **kwargs,
        )
        if not isinstance(output_kwargs_raw, dict):
            raise RuntimeError("Processor kwargs merge returned an invalid structure.")

        text_kwargs: dict[str, object] = {"padding": False, "padding_side": "left"}
        text_kwargs_raw = output_kwargs_raw.get("text_kwargs", {})
        if not isinstance(text_kwargs_raw, dict):
            raise TypeError("Merged `text_kwargs` must be a dictionary.")
        for key, value in text_kwargs_raw.items():
            if not isinstance(key, str):
                raise TypeError("`text_kwargs` keys must be strings.")
            text_kwargs[key] = value

        common_kwargs_raw = output_kwargs_raw.get("common_kwargs", {})
        return_tensors: object = None
        if isinstance(common_kwargs_raw, dict):
            return_tensors = common_kwargs_raw.get("return_tensors")
        texts_inputs = tokenizer(text_inputs, **text_kwargs)
        tensor_type = return_tensors if isinstance(return_tensors, str) else None

        return BatchFeature(
            data={**texts_inputs},
            tensor_type=tensor_type,
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        tokenizer = self._require_tokenizer()
        return tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        tokenizer = self._require_tokenizer()
        return tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer = self._require_tokenizer()
        tokenizer_input_names = tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names))


__all__ = ["Qwen3TTSProcessor"]
