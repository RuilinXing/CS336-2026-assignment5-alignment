import gzip
import importlib.util
import json
from pathlib import Path

from cs336_alignment.safety_rlhf import load_hh_preference_dataset


def _load_evaluation_script():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_safety_benchmarks.py"
    spec = importlib.util.spec_from_file_location("evaluate_safety_benchmarks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
