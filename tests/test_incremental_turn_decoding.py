import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

from qwen_tts.core.models.modeling_qwen3_tts_base import (
    Qwen3TTSConditionalGenerationBase,
)
from qwen_tts.inference.qwen3_tts_base_model import Qwen3TTSBaseModel


class _FakeDecodeStream:
    def __init__(self, calls: list[torch.Tensor]) -> None:
        self.calls = calls

    def decode_chunk(self, codes: torch.Tensor) -> tuple[np.ndarray, int]:
        self.calls.append(codes.detach().clone())
        return np.zeros(int(codes.shape[0]) * 4, dtype=np.float32), 24_000


class _FakeSpeechTokenizer:
    def __init__(self) -> None:
        self.stream_calls: list[list[torch.Tensor]] = []

    @staticmethod
    def supports_incremental_decode() -> bool:
        return True

    def create_decode_stream(self) -> _FakeDecodeStream:
        calls: list[torch.Tensor] = []
        self.stream_calls.append(calls)
        return _FakeDecodeStream(calls)

    @staticmethod
    def get_decode_upsample_rate() -> int:
        return 4


class IncrementalTurnDecodingTest(unittest.TestCase):
    def test_prefix_primes_one_state_and_turns_decode_only_fresh_codes(self) -> None:
        tokenizer = _FakeSpeechTokenizer()
        wrapper = object.__new__(Qwen3TTSBaseModel)
        wrapper.model = cast(
            Qwen3TTSConditionalGenerationBase,
            SimpleNamespace(speech_tokenizer=tokenizer),
        )

        prefix = torch.full((2, 2), 1, dtype=torch.long)
        first = torch.full((3, 2), 2, dtype=torch.long)
        second = torch.full((5, 2), 3, dtype=torch.long)
        wavs, sample_rate = wrapper._decode_talker_turns(
            [first, second], prefix_code=prefix
        )

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual([wav.shape[0] for wav in wavs], [12, 20])
        self.assertEqual(len(tokenizer.stream_calls), 1)
        calls = tokenizer.stream_calls[0]
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(calls[0], prefix)
        torch.testing.assert_close(calls[1], first)
        torch.testing.assert_close(calls[2], second)

        wrapper._decode_talker_turns([first])
        self.assertEqual(len(tokenizer.stream_calls), 2)
        self.assertEqual(len(tokenizer.stream_calls[1]), 1)


if __name__ == "__main__":
    unittest.main()
