# coding=utf-8
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .base import Qwen3TTSModelBase
from .voice_design_single_streamer import VoiceDesignSingleStreamer


class Qwen3TTSVoiceDesign(Qwen3TTSModelBase):
    """Standalone wrapper for Qwen3 TTS VoiceDesign model: generate_voice_design, generate_voice_design_single, stream_voice_design_single."""

    def _resolve_voice_design_language_id(self, language: Optional[str]) -> Tuple[str, Optional[int]]:
        lang = language if language is not None else "Auto"
        self._validate_languages([lang])
        if lang.lower() == "auto":
            return lang, None

        codec_lang = self.model.config.talker_config.codec_language_id
        language_id = codec_lang.get(lang.lower())
        if language_id is None:
            raise ValueError(
                f"Language {lang} not in codec_language_id. Supported: {sorted(codec_lang.keys())}"
            )
        return lang, language_id

    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`,
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    @torch.no_grad()
    def generate_voice_design_single(
        self,
        text: str,
        instruct: str = "",
        language: Optional[str] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech for a single sample with the VoiceDesign model (no batching).

        Uses the model's generate_single_voice_design path to avoid batch padding and list handling.
        Same semantics as generate_voice_design but accepts single str arguments and
        returns a single waveform and sample rate.

        Args:
            text: Text to synthesize.
            instruct: Instruction describing desired voice/style. Empty string = no instruction.
            language: Language for the sample. None or "Auto" for auto-detect.
            non_streaming_mode: Same as generate_voice_design.
            **kwargs: Generation options (do_sample, top_k, top_p, temperature, etc.),
                forwarded to model.generate_single_voice_design().

        Returns:
            Tuple[np.ndarray, int]: (waveform, sample_rate).
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design_single. Use a VoiceDesign model."
            )

        _, language_id = self._resolve_voice_design_language_id(language)

        input_ids = self._tokenize_texts([self._build_assistant_text(text)])[0]
        instruct_ids: Optional[torch.Tensor] = None
        if instruct is not None and instruct != "":
            instruct_ids = self._tokenize_texts([self._build_instruct_text(instruct)])[0]

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes, _ = self.model.generate_single_voice_design(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language_id=language_id,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": talker_codes}])
        return wavs[0], fs

    @torch.no_grad()
    def stream_voice_design_single(
        self,
        language: Optional[str] = None,
        instruct: str = "",
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> VoiceDesignSingleStreamer:
        """
        Create a stateful VoiceDesign streamer for incremental text->audio calls.

        The returned streamer keeps generation context across `stream()` calls.
        Each `stream()` call generates audio for all the text tokens it received.
        Use `streamer.finalize()` (or `streamer.flush()`) after the last text chunk
        to drain remaining audio.
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support stream_voice_design_single. Use a VoiceDesign model."
            )
        if non_streaming_mode:
            raise ValueError(
                "stream_voice_design_single requires non_streaming_mode=False. "
                "non_streaming_mode=True disables incremental text conditioning."
            )

        lang, language_id = self._resolve_voice_design_language_id(language)
        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        instruct_ids: Optional[torch.Tensor] = None
        if instruct is not None and instruct != "":
            instruct_ids = self._tokenize_texts([self._build_instruct_text(instruct)])[0]

        return VoiceDesignSingleStreamer(
            tts_model=self,
            language=lang,
            language_id=language_id,
            instruct=instruct,
            instruct_ids=instruct_ids,
            non_streaming_mode=non_streaming_mode,
            generate_kwargs=gen_kwargs,
        )
