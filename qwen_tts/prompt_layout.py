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
"""
Shared token-layout constants for prompt templates used by Qwen3-TTS.

These values describe the tokenized layout produced by:
  - _build_assistant_text()
  - _build_ref_text()
"""

# "<|im_start|>assistant\n" prefix.
ASSISTANT_PREFIX_TOKEN_COUNT = 3

# First text token starts immediately after the assistant prefix.
ASSISTANT_TEXT_START_TOKEN_INDEX = ASSISTANT_PREFIX_TOKEN_COUNT

# Trailing text conditioning skips the first text token in streaming mode.
ASSISTANT_TRAILING_TEXT_START_TOKEN_INDEX = ASSISTANT_TEXT_START_TOKEN_INDEX + 1

# "<|im_end|>\n<|im_start|>assistant\n" suffix used by _build_assistant_text().
ASSISTANT_SUFFIX_TOKEN_COUNT = 5

# Open-ended assistant prompt used by incremental streaming.
ASSISTANT_OPEN_SUFFIX_TOKEN_COUNT = 0

# "<|im_end|>\n" suffix used by _build_ref_text().
REFERENCE_SUFFIX_TOKEN_COUNT = 2

__all__ = [
    "ASSISTANT_PREFIX_TOKEN_COUNT",
    "ASSISTANT_TEXT_START_TOKEN_INDEX",
    "ASSISTANT_TRAILING_TEXT_START_TOKEN_INDEX",
    "ASSISTANT_SUFFIX_TOKEN_COUNT",
    "ASSISTANT_OPEN_SUFFIX_TOKEN_COUNT",
    "REFERENCE_SUFFIX_TOKEN_COUNT",
]
