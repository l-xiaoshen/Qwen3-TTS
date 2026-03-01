"""Stream VoiceDesign single-sample demo.

Shows how to pipe word-level LLM token output into the TTS streamer
so audio generation begins before the full text is available.
"""

import os
import pathlib
import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def llm_text_stream():
    """Simulate ~12-token batched streaming from an upstream LLM."""
    batches = [
        "Artificial intelligence has transformed the",
        " way we interact with technology.",
        " From virtual assistants that understand natural language to",
        " recommendation systems that predict our preferences,",
        " AI is woven into the fabric of modern life.\n",
        "One of the most exciting developments in recent years",
        " is the rise of large language models.",
        " These models can generate coherent text, translate between",
        " languages, write code, and even compose poetry.",
        " They learn from vast amounts of data and develop",
        " an understanding of grammar, context, and meaning.\n",
        "Text-to-speech technology has also made remarkable progress.",
        " Modern TTS systems produce voices that sound natural",
        " and expressive, capturing nuances like emotion, emphasis,",
        " and rhythm. When combined with streaming capabilities,",
        " these systems can begin speaking almost instantly,",
        " delivering a seamless conversational experience.",
    ]
    for batch in batches:
        time.sleep(0.03 * 12)
        yield batch


def main():
    model_path = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    device = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
    is_cpu = str(device).startswith("cpu")
    attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPL", "eager" if is_cpu else "sdpa")
    dtype_name = os.environ.get("QWEN3_TTS_DTYPE", "float32" if is_cpu else "bfloat16")
    dtype = getattr(torch, dtype_name)

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    streamer = model.stream_voice_design_single(
        language="English",
        instruct="A clear and natural male conversational voice.",
    )

    out_dir = pathlib.Path("output")
    out_dir.mkdir(exist_ok=True)

    wav_chunks = []
    chunk_idx = 0
    t0 = time.time()

    for idx, token in enumerate(llm_text_stream()):
        wav = streamer.stream(token)
        if wav.size > 0:
            wav_chunks.append(wav)
            sf.write(out_dir / f"chunk_{chunk_idx:03d}.wav", wav, streamer.sample_rate)
            chunk_idx += 1
        elapsed = time.time() - t0
        print(f"[{elapsed:5.2f}s] token={idx:03d} text={token!r} audio_samples={len(wav)}")

    while True:
        wav = streamer.finalize()
        if wav.size == 0:
            break
        wav_chunks.append(wav)
        sf.write(out_dir / f"chunk_{chunk_idx:03d}.wav", wav, streamer.sample_rate)
        chunk_idx += 1

    sr = streamer.sample_rate
    streamer.close()

    total = time.time() - t0
    if wav_chunks:
        merged = np.concatenate(wav_chunks, axis=0).astype(np.float32)
        merged_path = out_dir / "merged.wav"
        sf.write(merged_path, merged, sr)
        print(f"Saved {merged_path} — "
              f"chunks={chunk_idx}, samples={len(merged)}, sr={sr}, total={total:.2f}s")
    else:
        print("No audio generated.")


if __name__ == "__main__":
    main()
