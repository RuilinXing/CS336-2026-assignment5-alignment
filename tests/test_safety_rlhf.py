import gzip
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from cs336_alignment.gsm8k import load_gsm8k_examples
from cs336_alignment.safety_rlhf import load_hh_preference_dataset


def _load_evaluation_script():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_safety_benchmarks.py"
    spec = importlib.util.spec_from_file_location("evaluate_safety_benchmarks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_grpo_script():
    script_path = Path(__file__).parents[1] / "scripts" / "grpo.py"
    spec = importlib.util.spec_from_file_location("grpo_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_hh_preference_dataset_filters_multi_turn_conversations(tmp_path):
    single_turn = {
        "chosen": "\n\nHuman: Say hello.\n\nAssistant: Hello!",
        "rejected": "\n\nHuman: Say hello.\n\nAssistant: No.",
    }
    multi_turn = {
        "chosen": "\n\nHuman: Say hello.\n\nAssistant: Hello!\n\nHuman: Again.\n\nAssistant: Hello again!",
        "rejected": "\n\nHuman: Say hello.\n\nAssistant: No.\n\nHuman: Again.\n\nAssistant: No.",
    }
    for filename in (
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz",
    ):
        with gzip.open(tmp_path / filename, "wt", encoding="utf-8") as output_file:
            output_file.write(json.dumps(single_turn) + "\n")
            output_file.write(json.dumps(multi_turn) + "\n")

    examples = load_hh_preference_dataset(tmp_path)

    assert len(examples) == 4
    assert {example.source for example in examples} == {
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    }
    assert examples[0].instruction == "Say hello."
    assert examples[0].response_chosen == "Hello!"
    assert examples[0].response_rejected == "No."


def test_evaluation_serializers_write_json_and_jsonl(tmp_path):
    evaluation_script = _load_evaluation_script()
    summary_path = tmp_path / "summary.json"
    alpaca_path = tmp_path / "alpaca.json"
    safety_path = tmp_path / "safety.jsonl"

    evaluation_script._write_json(summary_path, {"records": [{"prediction": "A"}]})
    evaluation_script._write_json(alpaca_path, [{"instruction": "Hi", "output": "Hello"}])
    evaluation_script._write_jsonl(safety_path, [{"prompts_final": "Hi", "output": "Hello"}])

    assert json.loads(summary_path.read_text(encoding="utf-8"))["records"][0]["prediction"] == "A"
    assert isinstance(json.loads(alpaca_path.read_text(encoding="utf-8")), list)
    assert json.loads(safety_path.read_text(encoding="utf-8"))["output"] == "Hello"


def test_question_only_prompt_uses_its_matching_reward_and_no_r1_stop_sequence():
    grpo_script = _load_grpo_script()
    prompt_path = Path(__file__).parents[1] / "cs336_alignment" / "prompts" / "question_only.prompt"

    reward_fn, stop_sequences = grpo_script.resolve_rollout_configuration(prompt_path)
    reward = reward_fn(r"\boxed{18}", "18")

    assert reward["reward"] == 1.0
    assert stop_sequences is None


def test_custom_rollout_prompt_requires_an_explicit_format(tmp_path):
    grpo_script = _load_grpo_script()
    custom_prompt_path = tmp_path / "math_prompt.prompt"

    with pytest.raises(ValueError, match="rollout-format"):
        grpo_script.resolve_rollout_configuration(custom_prompt_path)

    _, stop_sequences = grpo_script.resolve_rollout_configuration(
        custom_prompt_path,
        rollout_format="question_only",
    )
    assert stop_sequences is None


def test_standard_grpo_rejects_partial_rollout_training_batches():
    grpo_script = _load_grpo_script()
    args = Namespace(
        rollout_batch_size=256,
        group_size=8,
        num_rollout_steps=1,
        eval_every=1,
    )

    with pytest.raises(ValueError, match="train_batch_size"):
        grpo_script.validate_args(
            args,
            grpo_script.ALGORITHM_SETTINGS["standard"],
            train_batch_size=128,
            gradient_accumulation_steps=32,
            updates_per_rollout=1,
        )


@pytest.mark.parametrize("group_size", [0, -1])
def test_grpo_rejects_nonpositive_group_size(group_size):
    grpo_script = _load_grpo_script()
    args = Namespace(
        rollout_batch_size=256,
        group_size=group_size,
        num_rollout_steps=1,
        eval_every=1,
    )

    with pytest.raises(ValueError, match="group_size"):
        grpo_script.validate_args(
            args,
            grpo_script.ALGORITHM_SETTINGS["standard"],
            train_batch_size=256,
            gradient_accumulation_steps=32,
            updates_per_rollout=1,
        )


def test_numeric_gsm8k_answers_are_compared_by_value():
    evaluation_script = _load_evaluation_script()

    assert evaluation_script.numeric_answers_equal("18.0", "18")
    assert evaluation_script.numeric_answers_equal("1,000", "1000")
    assert not evaluation_script.numeric_answers_equal("18.5", "18")


def test_validation_response_length_uses_generated_tokens():
    grpo_script = _load_grpo_script()

    class StubServer:
        def generate_completions(self, prompts, sampling_params, batch_size):
            del prompts, sampling_params, batch_size
            return [
                grpo_script.VLLMCompletion(
                    text="one hundred characters are not required",
                    token_ids=[1, 2, 3],
                    finish_reason="stop",
                )
            ]

    metrics, _ = grpo_script.evaluate_policy(
        StubServer(),
        [grpo_script.GSM8KExample(question="What is 1 + 1?", ground_truth="2")],
        "Question: {question}",
        {"temperature": 0.0},
        1,
        lambda response, ground_truth: {
            "reward": float(response and ground_truth),
            "format_reward": 1.0,
            "answer_reward": 1.0,
        },
    )

    assert metrics["val_average_response_length"] == 3.0


def test_generate_grouped_responses_requests_multiple_samples_per_unique_prompt():
    grpo_script = _load_grpo_script()

    class StubServer:
        def __init__(self):
            self.prompts = None
            self.sampling_params = None
            self.batch_size = None

        def generate_completions(self, prompts, sampling_params, batch_size):
            self.prompts = prompts
            self.sampling_params = sampling_params
            self.batch_size = batch_size
            return [
                grpo_script.VLLMCompletion(text=f"sample-{index}", token_ids=[index], finish_reason="stop")
                for index in range(len(prompts) * sampling_params["n"])
            ]

    server = StubServer()
    completions = grpo_script.generate_grouped_responses(
        server,
        ["prompt-a", "prompt-b"],
        group_size=3,
        sampling_params={"n": 1, "seed": 17},
        batch_size=6,
    )

    assert server.prompts == ["prompt-a", "prompt-b"]
    assert server.sampling_params == {"n": 3, "seed": 17}
    assert server.batch_size == 6
    assert [completion.text for completion in completions] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
        "sample-5",
    ]


def test_gsm8k_loader_respects_zero_example_limit(tmp_path):
    dataset_path = tmp_path / "gsm8k.jsonl"
    dataset_path.write_text(
        json.dumps({"question": "What is 1 + 1?", "answer": "1 + 1 = 2\\n#### 2"}) + "\n",
        encoding="utf-8",
    )

    assert load_gsm8k_examples(dataset_path, max_examples=0) == []
