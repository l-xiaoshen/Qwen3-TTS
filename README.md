# Qwen3-TTS Streaming Fork

This fork of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) adds streaming text-to-speech generation. It accepts text tokens incrementally as they arrive from a language model and returns audio in chunks. Audio generation begins before the full text is available.

For installation, environment setup, and all other features, see the [upstream repository](https://github.com/QwenLM/Qwen3-TTS).

## What this fork adds

- A stateful streamer that maintains decoder state across calls, allowing token-by-token text input with incremental audio output.
- A single-sample generation path that avoids batch padding overhead when only one utterance is needed.
- Core model methods for single-sample streaming with KV-cache reuse across generation steps.

All upstream batch APIs remain unchanged.

## Usage

See [examples/stream_voice_design_single_demo.py](examples/stream_voice_design_single_demo.py) for a complete working example.

The streaming workflow has five steps:

1. Load the VoiceDesign model.
2. Create a streamer by calling `stream_voice_design_single()` with a language and a voice description.
3. For each text chunk from the language model (single tokens or batches), call `streamer.stream()`. It returns an audio array. If the array is not empty, the audio is ready for playback or storage.
4. After the last text token, call `streamer.finalize()` repeatedly until it returns an empty array.
5. Call `streamer.close()` to release resources.

For single-sample non-streaming generation, call `generate_voice_design_single()` with the full text, language, and voice description.

## How it works

The streamer maintains an open-ended prompt prefix that grows as text chunks arrive. On each call to `stream()`, new tokens are appended to the cached prefix and passed directly to the model without any framing suffix. The model generates audio codes for the available text with end-of-sequence suppressed so that generation can continue later. Only newly generated waveform samples are returned.

When `finalize()` is called, end-of-sequence suppression is removed so that the model can properly terminate generation.

## Upstream

Based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team.

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
