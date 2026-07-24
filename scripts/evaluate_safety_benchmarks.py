"""Evaluate supplement benchmarks with zero-shot or SFT prompt formatting."""

from __future__ import annotations

import argparse
import csv
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from cs336_alignment.safety_rlhf import (
    parse_gsm8k_response,
    parse_mmlu_response,
    render_alpaca_sft,
)
from cs336_alignment.vllm_utils import VLLMServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "cs336_alignment" / "prompts_safety"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("mmlu", "gsm8k", "alpaca_eval", "simple_safety_tests"), required=True)
    parser.add_argument("--prompt-mode", choices=("zero-shot", "sft"), required=True)
    parser.add_argument("--model-id", required=True, help="Hugging Face model ID or local model path.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mmlu-dir", type=Path, default=PROJECT_ROOT / "data" / "mmlu" / "test")
    parser.add_argument("--gsm8k-path", type=Path, default=PROJECT_ROOT / "data" / "gsm8k" / "test.jsonl")
    parser.add_argument("--alpaca-eval-path", type=Path, default=PROJECT_ROOT / "data" / "alpaca_eval" / "alpaca_eval_gpt4_turbo.json")
    parser.add_argument("--safety-tests-path", type=Path, default=PROJECT_ROOT / "data" / "simple_safety_tests" / "simple_safety_tests.csv")
    return parser.parse_args()


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _wrap_prompt(instruction: str, prompt_mode: str) -> str:
    if prompt_mode == "sft":
        return render_alpaca_sft(instruction)
    return _read_prompt("zero_shot_system_prompt.prompt").format(instruction=instruction)


def _task_instruction(task: str, example: dict[str, Any]) -> str:
    if task == "mmlu":
        return _read_prompt("mmlu_zero_shot.prompt").format(**example)
    if task == "gsm8k":
        return _read_prompt("gsm8k_zero_shot.prompt").format(question=example["question"])
    if task == "alpaca_eval":
        return _read_prompt("alpaca_eval_zero_shot.prompt").format(instruction=example["instruction"])
    return _read_prompt("simple_safety_tests_zero_shot.prompt").format(
        instruction=example["prompts_final"]
    )


def load_mmlu_examples(dataset_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.glob("*_test.csv")):
        subject = path.stem.removesuffix("_test").replace("_", " ")
        with path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.reader(csv_file):
                if len(row) != 6:
                    raise ValueError(f"Expected six MMLU columns in {path}, got {len(row)}.")
                examples.append(
                    {
                        "subject": subject,
                        "question": row[0],
                        "options": row[1:5],
                        "answer": row[5],
                    }
                )
    return examples


def load_gsm8k_examples(dataset_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            examples.append(
                {
                    "question": record["question"],
                    "answer": record["answer"].split("####")[-1].strip().replace(",", ""),
                }
            )
    return examples


def load_alpaca_examples(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def load_safety_examples(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def numeric_answers_equal(prediction: str | None, answer: str) -> bool:
    """Compare parsed numeric answers while accepting equivalent decimal spellings."""
    if prediction is None:
        return False
    try:
        return Decimal(prediction.replace(",", "")) == Decimal(answer.replace(",", ""))
    except InvalidOperation:
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    loaders = {
        "mmlu": lambda: load_mmlu_examples(args.mmlu_dir),
        "gsm8k": lambda: load_gsm8k_examples(args.gsm8k_path),
        "alpaca_eval": lambda: load_alpaca_examples(args.alpaca_eval_path),
        "simple_safety_tests": lambda: load_safety_examples(args.safety_tests_path),
    }
    examples = loaders[args.task]()
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    prompts = [_wrap_prompt(_task_instruction(args.task, example), args.prompt_mode) for example in examples]
    server = VLLMServer(model_id=args.model_id, gpu=args.gpu)
    server.start()
    try:
        started_at = time.perf_counter()
        completions = server.generate_completions(
            prompts,
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "n": 1,
                "seed": args.seed,
                "stop": ["# Query:"] if args.prompt_mode == "zero-shot" else None,
            },
            batch_size=args.batch_size,
        )
        elapsed_seconds = time.perf_counter() - started_at
    finally:
        server.stop()

    if len(completions) != len(examples):
        raise RuntimeError("vLLM returned a different number of completions than prompts.")
    generations = [completion.text for completion in completions]

    if args.task == "alpaca_eval":
        records = [
            {
                "instruction": example["instruction"],
                "output": generation,
                "generator": args.model_id,
                "dataset": example["dataset"],
            }
            for example, generation in zip(examples, generations, strict=True)
        ]
        _write_json(args.output_path, records)
        return

    if args.task == "simple_safety_tests":
        records = [
            {**example, "output": generation}
            for example, generation in zip(examples, generations, strict=True)
        ]
        _write_jsonl(args.output_path, records)
        return

    parser = parse_mmlu_response if args.task == "mmlu" else lambda _example, text: parse_gsm8k_response(text)
    records = []
    for example, prompt, completion in zip(examples, prompts, completions, strict=True):
        prediction = parser(example, completion.text)
        records.append(
            {
                **example,
                "prompt": prompt,
                "generation": completion.text,
                "finish_reason": completion.finish_reason,
                "prediction": prediction,
                "correct": (
                    prediction == example["answer"]
                    if args.task == "mmlu"
                    else numeric_answers_equal(prediction, example["answer"])
                ),
            }
        )
    parsed_records = [record for record in records if record["prediction"] is not None]
    _write_json(
        args.output_path,
        {
            "task": args.task,
            "prompt_mode": args.prompt_mode,
            "model_id": args.model_id,
            "num_examples": len(records),
            "elapsed_seconds": elapsed_seconds,
            "examples_per_second": len(records) / elapsed_seconds if elapsed_seconds else None,
            "accuracy": sum(record["correct"] for record in records) / len(records) if records else None,
            "parse_failures": len(records) - len(parsed_records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
