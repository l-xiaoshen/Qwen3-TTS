# Qwen TTS Fork

This fork extends `qwen-tts` with additional inference APIs for speaker embeddings, reusable prompts, weighted speaker mixing, hybrid custom voice generation, and ordered multi-part `TTSInput` requests.

The fork keeps `voice_clone` and `custom_voice` as separate APIs:

- Use `Qwen3TTSVoiceCloneModel` for reference-based voice cloning.
- Use `Qwen3TTSCustomVoiceModel` for custom voice generation, speaker merge, and instruction control.
- Use the hybrid `custom_voice` flow when instruction control and reference prompting are both required.

## Features/API added in this fork

- Speaker embedding extraction.
- Separate single and batch generation APIs.
- Public `SpeakerConfiguration` support.
- Weighted speaker merge.
- Reusable prompt builders.
- Hybrid custom voice generation with `speaker`, reference audio, and reference text.
- Ordered `TTSInput` requests with one audio output per input part.

## Usage

### Build a `TTSInput`

Runtime generation now uses an ordered list of parts:

```python
tts_input = [
    {
        "instruction": "Speak slowly.",
        "text": "Hi folks, today",
    },
    {
        "instruction": "Speak much faster.",
        "text": "I'm going to introduce a new feature.",
    },
]
```

One `TTSInput` request is synthesized as one continuous generation stream. When `non_streaming_mode=False`, the runtime API returns one audio item per input part, and batch APIs return one list of part audios per request. When `non_streaming_mode=True`, multi-part requests are still accepted but may return a single output item for the full request.

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
wavs, sr = tts.generate_custom_voice(
    tts_input=tts_input,
    speaker=embedding,
)
```

For batch generation, use the batch API.

```python
tts_inputs = [tts_input, tts_input]
audio_batches, sr = tts.generate_custom_voice_batch(
    tts_inputs=tts_inputs,
    speaker=embeddings,
)
```

### Merge speakers

Use `SpeakerConfiguration` to mix built-in speakers.

```python
speaker = {"Vivian": 1.0, "Ryan": 0.3}
wavs, sr = tts.generate_custom_voice(tts_input=tts_input, speaker=speaker)
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
wavs, sr = clone.generate_voice_clone(tts_input=tts_input, voice_clone_prompt=prompt)
```

### Use reference prompting with instruction

Use the hybrid `custom_voice` flow when instruction control and reference prompting must be combined.

```python
wavs, sr = tts.generate_custom_voice(
    tts_input=tts_input,
    speaker=embedding,
    ref_audio=ref_audio,
    ref_text=ref_text,
)
```

For repeated requests, precompute the prompt once and reuse it.

```python
prompt = tts.create_custom_voice_prompt(
    ref_audio=[ref_audio],
    ref_text=[ref_text],
)[0]
wavs, sr = tts.generate_custom_voice(
    tts_input=tts_input,
    speaker=embedding,
    custom_voice_prompt=prompt,
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
- `TTSInputPart`
- `TTSInput`
- `VoiceClonePromptItem`
- `CustomVoicePromptItem`
- `SpeakerConfiguration`
