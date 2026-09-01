import unittest
from collections.abc import Callable

import torch
from typing_extensions import override

from qwen_tts.core.models.generation.single import Qwen3TTSGenerationSingleMixin
from qwen_tts.core.models.modeling_qwen3_tts_types import SubTalkerConfiguration


def _marker(value: int) -> torch.Tensor:
    return torch.tensor([[[float(value)]]])


class _GenerationHarness(Qwen3TTSGenerationSingleMixin):
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.turn_index = 0

    @override
    def _append_instruct_embed_block(
        self,
        talker_input_embeds: list[torch.Tensor],
        instruct_id: torch.Tensor | None,
    ) -> None:
        if instruct_id is not None:
            talker_input_embeds.append(instruct_id)

    @override
    def _build_talker_suppress_tokens(
        self, eos_token_id: int | None = None
    ) -> list[int]:
        return []

    @override
    def _run_talker_generation(
        self,
        talker_input_embeds: list[torch.Tensor],
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        suppress_tokens: list[int],
        max_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        subtalker_configuration: SubTalkerConfiguration | None,
        eos_token_id: int | None,
        repetition_penalty: float,
        codec_frame_callback: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        _ = (
            trailing_text_hidden,
            tts_pad_embed,
            suppress_tokens,
            max_new_tokens,
            do_sample,
            top_k,
            top_p,
            temperature,
            subtalker_configuration,
            eos_token_id,
            repetition_penalty,
            codec_frame_callback,
        )
        self.calls.append([int(block.item()) for block in talker_input_embeds])
        self.turn_index += 1
        codes = torch.full((2, 2), self.turn_index, dtype=torch.long)
        hidden = torch.zeros((2, 1))
        return codes, hidden, True

    @override
    def _build_generated_codec_history_embeddings(
        self,
        talker_codes: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
    ) -> torch.Tensor:
        _ = trailing_text_hidden, tts_pad_embed
        return _marker(100 + int(talker_codes[0, 0].item()))

    @override
    def _build_codec_eos_history_embedding(
        self,
        tts_pad_embed: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        generated_length: int,
        input_dtype: torch.dtype,
        eos_token_id: int,
    ) -> torch.Tensor:
        _ = (
            tts_pad_embed,
            trailing_text_hidden,
            generated_length,
            input_dtype,
            eos_token_id,
        )
        return _marker(200 + self.turn_index)


class MultiTurnContextTest(unittest.TestCase):
    def test_prior_instructions_text_audio_and_eos_remain_in_context(self) -> None:
        generation = _GenerationHarness()
        input_ids = [torch.ones(1, dtype=torch.long) for _ in range(3)]
        instructions = [_marker(11), None, _marker(13)]
        prompts = [
            (_marker(21), _marker(31), _marker(41)),
            (_marker(22), _marker(32), _marker(42)),
            (_marker(23), _marker(33), _marker(43)),
        ]

        generation._generate_turns_with_prepared_prompts(
            input_ids=input_ids,
            instruct_ids=instructions,
            prepared_prompts=prompts,
            max_new_tokens=10,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            subtalker_configuration=None,
            eos_token_id=9,
            repetition_penalty=1.0,
        )

        self.assertEqual(generation.calls[0], [11, 21])
        self.assertEqual(generation.calls[1], [11, 21, 101, 201, 22])
        self.assertEqual(
            generation.calls[2],
            [11, 21, 101, 201, 22, 102, 202, 13, 23],
        )


if __name__ == "__main__":
    unittest.main()
