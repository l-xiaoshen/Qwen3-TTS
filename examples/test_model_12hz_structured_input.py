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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSCustomVoiceModel, TTSInput

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = Path("qwen3_tts_structured_input_output")


def main() -> None:
    tts = Qwen3TTSCustomVoiceModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Each item is a user instruction followed by an assistant text/audio turn.
    # Later turns retain the earlier turns' text and generated codec history.
    tts_input: TTSInput = [
        {
            "text": "The house was completely silent when",
            "instruction": "Express caution and tension. Speak extreme slowly and softly.",
            #    "instruction": "Speak quickly with rising intensity, ANGRY.",
        },
        {
            "text": " I opened the door. Then every alarm in the building went off at once!",
            "instruction": "Speak extreme quickly with rising intensity.",
        },
        {
            "text": "It was only a drill, so we could finally laugh about it.",
            "instruction": "Speak slowly and softly.",
            # "instruction": ""
        },
    ]

    turn_wavs, sample_rate = tts.generate_custom_voice(
        tts_input=tts_input,
        language="English",
        speaker={"Ryan": 1.0},
        # non_streaming_mode=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, wav in enumerate(turn_wavs, start=1):
        output_path = OUTPUT_DIR / f"turn_{index:02d}.wav"
        sf.write(output_path, wav, sample_rate)
        print(f"Saved {output_path}")

    pause = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    preview_parts: list[np.ndarray] = []
    for index, wav in enumerate(turn_wavs):
        preview_parts.append(wav)
        if index + 1 < len(turn_wavs):
            preview_parts.append(pause)
    preview_path = OUTPUT_DIR / "shared_context_preview.wav"
    sf.write(preview_path, np.concatenate(preview_parts), sample_rate)
    print(f"Saved {preview_path}")


if __name__ == "__main__":
    main()
