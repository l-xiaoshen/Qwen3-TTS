import inspect
import unittest
from unittest.mock import patch

import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSCustomVoiceForConditionalGeneration,
    Qwen3TTSProcessor,
    Qwen3TTSVoiceCloneForConditionalGeneration,
    Qwen3TTSVoiceDesignForConditionalGeneration,
)
from qwen_tts.inference.qwen3_tts_base_model import (
    GenerationDefaults,
    Qwen3TTSBaseModel,
)
from qwen_tts.inference.qwen3_tts_custom_voice_model import Qwen3TTSCustomVoiceModel
from qwen_tts.inference.qwen3_tts_voice_clone_model import Qwen3TTSVoiceCloneModel
from qwen_tts.inference.qwen3_tts_voice_design_model import Qwen3TTSVoiceDesignModel


class GenerationApiTest(unittest.TestCase):
    wrapper_methods = (
        Qwen3TTSCustomVoiceModel.generate_custom_voice,
        Qwen3TTSCustomVoiceModel.generate_custom_voice_batch,
        Qwen3TTSVoiceCloneModel.generate_voice_clone,
        Qwen3TTSVoiceCloneModel.generate_voice_clone_batch,
        Qwen3TTSVoiceDesignModel.generate_voice_design,
        Qwen3TTSVoiceDesignModel.generate_voice_design_batch,
    )
    core_methods = (
        Qwen3TTSCustomVoiceForConditionalGeneration.generate_custom_voice,
        Qwen3TTSCustomVoiceForConditionalGeneration.generate_custom_voice_turns,
        Qwen3TTSCustomVoiceForConditionalGeneration.generate_custom_voice_batch,
        Qwen3TTSVoiceCloneForConditionalGeneration.generate_voice_clone,
        Qwen3TTSVoiceCloneForConditionalGeneration.generate_voice_clone_turns,
        Qwen3TTSVoiceCloneForConditionalGeneration.generate_voice_clone_batch,
        Qwen3TTSVoiceDesignForConditionalGeneration.generate_voice_design,
        Qwen3TTSVoiceDesignForConditionalGeneration.generate_voice_design_turns,
        Qwen3TTSVoiceDesignForConditionalGeneration.generate_voice_design_batch,
    )

    def test_generation_methods_have_closed_signatures(self) -> None:
        for method in self.wrapper_methods + self.core_methods:
            with self.subTest(method=method.__qualname__):
                parameters = inspect.signature(method).parameters
                self.assertNotIn("kwargs", parameters)
                self.assertFalse(
                    any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                )
                self.assertNotIn("output_hidden_states", parameters)
                self.assertNotIn("return_dict_in_generate", parameters)
                self.assertEqual(
                    parameters["eos_token_id"].kind,
                    inspect.Parameter.KEYWORD_ONLY,
                )

    def test_wrapper_generation_overrides_default_to_none(self) -> None:
        override_names = (
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "subtalker_configuration",
            "max_new_tokens",
            "eos_token_id",
        )
        for method in self.wrapper_methods:
            parameters = inspect.signature(method).parameters
            with self.subTest(method=method.__qualname__):
                for name in override_names:
                    self.assertIsNone(parameters[name].default)

    def test_explicit_hard_defaults_override_checkpoint_defaults(self) -> None:
        model = object.__new__(Qwen3TTSBaseModel)
        model.generate_defaults = GenerationDefaults(
            do_sample=False,
            top_k=20,
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.2,
            max_new_tokens=1024,
            subtalker_configuration={"top_k": 12, "temperature": 0.6},
        )

        resolved = model._resolve_generation_options(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_configuration={"top_k": 50, "temperature": 0.9},
            max_new_tokens=2048,
            eos_token_id=42,
        )

        self.assertEqual(resolved["do_sample"], True)
        self.assertEqual(resolved["top_k"], 50)
        self.assertEqual(resolved["top_p"], 1.0)
        self.assertEqual(resolved["temperature"], 0.9)
        self.assertEqual(resolved["repetition_penalty"], 1.05)
        self.assertEqual(resolved["subtalker_configuration"].get("top_k"), 50)
        self.assertEqual(resolved["subtalker_configuration"].get("temperature"), 0.9)
        self.assertEqual(resolved["max_new_tokens"], 2048)
        self.assertEqual(resolved["eos_token_id"], 42)

    def test_omitted_overrides_use_checkpoint_defaults(self) -> None:
        model = object.__new__(Qwen3TTSBaseModel)
        model.generate_defaults = GenerationDefaults(
            do_sample=False,
            top_k=20,
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.2,
            max_new_tokens=1024,
            subtalker_configuration={"top_k": 12, "temperature": 0.6},
        )

        resolved = model._resolve_generation_options()

        self.assertEqual(resolved["do_sample"], False)
        self.assertEqual(resolved["top_k"], 20)
        self.assertEqual(resolved["top_p"], 0.8)
        self.assertEqual(resolved["temperature"], 0.7)
        self.assertEqual(resolved["repetition_penalty"], 1.2)
        self.assertEqual(resolved["subtalker_configuration"].get("do_sample"), True)
        self.assertEqual(resolved["subtalker_configuration"].get("top_k"), 12)
        self.assertEqual(resolved["subtalker_configuration"].get("top_p"), 1.0)
        self.assertEqual(resolved["subtalker_configuration"].get("temperature"), 0.6)
        self.assertEqual(resolved["max_new_tokens"], 1024)
        self.assertIsNone(resolved["eos_token_id"])

    def test_loader_preserves_model_kwargs_after_config_dispatch(self) -> None:
        config = Qwen3TTSConfig(tts_model_type="custom_voice")
        model = object.__new__(Qwen3TTSCustomVoiceForConditionalGeneration)
        object.__setattr__(model, "generate_config", None)
        processor = object.__new__(Qwen3TTSProcessor)

        with (
            patch.object(AutoConfig, "register"),
            patch.object(AutoProcessor, "register"),
            patch.object(
                AutoConfig, "from_pretrained", return_value=config
            ) as config_loader,
            patch.object(
                Qwen3TTSCustomVoiceForConditionalGeneration,
                "from_pretrained",
                return_value=model,
            ) as model_loader,
            patch.object(
                AutoProcessor, "from_pretrained", return_value=processor
            ) as processor_loader,
            patch.object(Qwen3TTSBaseModel, "__init__", return_value=None),
        ):
            Qwen3TTSBaseModel.from_pretrained(
                "model-id",
                dtype=torch.bfloat16,
                subfolder="nested",
            )

            self.assertEqual(processor_loader.call_args.kwargs["subfolder"], "nested")

            Qwen3TTSBaseModel.from_pretrained(
                "model-id",
                dtype=torch.bfloat16,
            )

            self.assertNotIn("subfolder", processor_loader.call_args.kwargs)

        nested_config_kwargs = config_loader.call_args_list[0].kwargs
        nested_model_kwargs = model_loader.call_args_list[0].kwargs
        self.assertEqual(nested_config_kwargs["dtype"], torch.bfloat16)
        self.assertEqual(nested_config_kwargs["subfolder"], "nested")
        self.assertIsNone(nested_model_kwargs["config"])
        self.assertEqual(nested_model_kwargs["dtype"], torch.bfloat16)
        self.assertEqual(nested_model_kwargs["subfolder"], "nested")


if __name__ == "__main__":
    unittest.main()
