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

import argparse
import json
from collections.abc import Sequence

from dataset import (
    PreparedTTSJsonRow,
    RawTTSJsonRow,
    add_audio_codes,
    parse_audio_codes,
    parse_raw_tts_json_row,
)

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32


class PrepareDataArgs(argparse.Namespace):
    device: str
    tokenizer_model_path: str
    input_jsonl: str
    output_jsonl: str


def _encode_batch(
    tokenizer: Qwen3TTSTokenizer,
    rows: list[RawTTSJsonRow],
    audios: list[str],
    output: list[PreparedTTSJsonRow],
) -> None:
    encoded = tokenizer.encode(audios)
    if len(encoded.audio_codes) != len(rows):
        raise RuntimeError(
            "Tokenizer returned a different number of code sequences than inputs."
        )
    for index, (code, row) in enumerate(zip(encoded.audio_codes, rows)):
        codes_raw: object = code.detach().cpu().tolist()
        codes = parse_audio_codes(codes_raw, f"tokenizer output[{index}]")
        output.append(add_audio_codes(row, codes))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args(argv, namespace=PrepareDataArgs())

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    with open(args.input_jsonl) as input_file:
        input_lines = input_file.readlines()
    total_lines: list[RawTTSJsonRow] = []
    for line_number, line in enumerate(input_lines, start=1):
        value: object = json.loads(line.strip())
        total_lines.append(
            parse_raw_tts_json_row(value, f"input JSONL line {line_number}")
        )

    final_lines: list[PreparedTTSJsonRow] = []
    batch_lines: list[RawTTSJsonRow] = []
    batch_audios: list[str] = []
    for line in total_lines:
        batch_lines.append(line)
        batch_audios.append(line["audio"])

        if len(batch_lines) >= BATCH_INFER_NUM:
            _encode_batch(tokenizer_12hz, batch_lines, batch_audios, final_lines)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        _encode_batch(tokenizer_12hz, batch_lines, batch_audios, final_lines)
        batch_lines.clear()
        batch_audios.clear()

    with open(args.output_jsonl, "w") as f:
        for line in final_lines:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
