# Qwen TTS Fork

This fork extends `qwen-tts` with additional inference APIs for speaker embeddings, reusable prompts, weighted speaker mixing, and hybrid custom voice generation.

The fork keeps `voice_clone` and `custom_voice` as separate APIs:

- Use `Qwen3TTSVoiceCloneModel` for reference-based voice cloning.
- Use `Qwen3TTSCustomVoiceModel` for custom voice generation, speaker merge, and instruction control.
- Use the hybrid `custom_voice` flow when instruction control and reference prompting are both required.

## Features added in this fork

- Speaker embedding extraction.
- Separate single and batch generation APIs.
- Public `SpeakerConfiguration` support.
- Weighted speaker merge.
- Reusable prompt builders.
- Hybrid custom voice generation with `speaker`, reference audio, reference text, and `instruct`.

`Qwen3TTSVoiceCloneModel` does not expose `instruct`. When instruction control is required, use `Qwen3TTSCustomVoiceModel.generate_custom_voice()` with a direct speaker embedding plus a reference prompt.

## Minimal API flows

### Extract a speaker embedding

Use the base model to extract a speaker embedding from reference audio.

```python
from qwen_tts import Qwen3TTSVoiceCloneModel

base = Qwen3TTSVoiceCloneModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
embedding = base.extract_speaker_embedding(ref_audio)
```

For batch extraction, use the batch API.

```python
embeddings = base.extract_speaker_embedding_batch(ref_audios)
```

### Generate custom voice

Use a direct speaker embedding or a weighted speaker configuration.

```python
from qwen_tts import Qwen3TTSCustomVoiceModel

tts = Qwen3TTSCustomVoiceModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
wav, sr = tts.generate_custom_voice(
    text=text,
    speaker=embedding,
    instruct="Calm and low voice.",
)
```

For batch generation, use the batch API.

```python
wavs, sr = tts.generate_custom_voice_batch(
    text=texts,
    speaker=embeddings,
    instruct=instructs,
)
```

### Merge speakers

Use `SpeakerConfiguration` to mix built-in speakers.

```python
speaker = {"Vivian": 1.0, "Ryan": 0.3}
wav, sr = tts.generate_custom_voice(text=text, speaker=speaker)
```

### Clone from reference audio and reference text

Use `voice_clone` when the flow is fully reference based.

```python
from qwen_tts import Qwen3TTSVoiceCloneModel

clone = Qwen3TTSVoiceCloneModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
prompt = clone.create_voice_clone_prompt(
    ref_audio=[ref_audio],
    ref_text=[ref_text],
)[0]
wav, sr = clone.generate_voice_clone(text=text, voice_clone_prompt=prompt)
```

### Use reference prompting with instruction

Use the hybrid `custom_voice` flow when instruction control and reference prompting must be combined.

```python
wav, sr = tts.generate_custom_voice(
    text=text,
    speaker=embedding,
    ref_audio=ref_audio,
    ref_text=ref_text,
    instruct="Speak with restrained frustration.",
)
```

For repeated requests, precompute the prompt once and reuse it.

```python
prompt = tts.create_custom_voice_prompt(
    ref_audio=[ref_audio],
    ref_text=[ref_text],
)[0]
wav, sr = tts.generate_custom_voice(
    text=text,
    speaker=embedding,
    custom_voice_prompt=prompt,
    instruct="Speak with restrained frustration.",
)
```

## Example files

The repository includes the following example flows:

- `examples/test_model_12hz_custom_voice.py`: Custom voice single and batch generation.
- `examples/test_model_12hz_voice_merge.py`: Weighted speaker merge with `SpeakerConfiguration`.
- `examples/test_model_12hz_base.py`: Voice clone single and batch generation, including prompt reuse.
- `examples/test_model_12hz_custom_voice_hybrid.py`: Hybrid custom voice generation with `speaker`, reference audio, reference text, and `instruct`.

## Public classes and prompt types

The fork exports the following top-level objects:

- `Qwen3TTSVoiceCloneModel`
- `Qwen3TTSCustomVoiceModel`
- `Qwen3TTSVoiceDesignModel`
- `VoiceClonePromptItem`
- `CustomVoicePromptItem`
- `SpeakerConfiguration`
