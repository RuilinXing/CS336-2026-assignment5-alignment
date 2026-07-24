"""
Evaluate OLMo prompting baselines on the GSM8K test split.

Running:

```
uv run python scripts/evaluate_prompting_baselines.py \\
    --output-path <path_to_write_results.json> \\
    --model-id "allenai/OLMo-2-0425-1B" \\
    --gpu 0
```

For a small debugging run, add `--max-examples 8`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.gsm8k import GSM8KExample, load_gsm8k_examples, render_prompt
from cs336_alignment.vllm_utils import VLLMServer

# 该路径用于定位仓库中的数据集和 prompt 文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 每种提示词格式都要使用匹配的奖励函数和停止字符串。
PROMPT_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "name": "question_only",
        "prompt_path": PROJECT_ROOT / "cs336_alignment/prompts/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    {
        "name": "r1_zero",
        "prompt_path": PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
    {
        "name": "r1_zero_three_shot_gsm8k",
        "prompt_path": PROJECT_ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path for the JSON experiment record.",
    )
    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-2-0425-1B",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "data/gsm8k/test.jsonl",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device for vLLM.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for a small debugging run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def reward_category(scores: dict[str, float]) -> str:
    format_reward = scores["format_reward"]
    answer_reward = scores["answer_reward"]
    if format_reward == 1.0 and answer_reward == 1.0:
        return "format_1_answer_1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format_1_answer_0"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "format_0_answer_0"
    return "other"


def evaluate_prompt(
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_config: dict[str, object],
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    prompt_template = Path(prompt_config["prompt_path"]).read_text(encoding="utf-8")
    prompts = [render_prompt(prompt_template, example.question) for example in examples]
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": seed,
        "stop": prompt_config["stop"],
        # R1 奖励函数需要读取结束标签，所以不能从输出中移除它。
        "include_stop_str_in_output": prompt_config["stop"] is not None,
    }
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    reward_fn: Callable[[str, str], dict[str, float]] = prompt_config["reward_fn"]  # type: ignore[assignment]
    records = []
    categories: Counter[str] = Counter()
    for example, prompt, completion in zip(examples, prompts, completions, strict=True):
        scores = reward_fn(completion.text, example.ground_truth)
        category = reward_category(scores)
        categories[category] += 1
        records.append(
            {
                "question": example.question,
                "ground_truth": example.ground_truth,
                "prompt": prompt,
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "scores": scores,
                "category": category,
            }
        )

    return {
        "prompt_name": prompt_config["name"],
        "counts": dict(categories),
        "records": records,
        # 作业要求人工检查两种解析失败情况，因此各保留十条样例。
        "manual_review": {
            category: [record for record in records if record["category"] == category][:10]
            for category in ("format_1_answer_0", "format_0_answer_0")
        },
    }


def main() -> None:
    args = parse_args()
    examples = load_gsm8k_examples(args.dataset_path, args.max_examples)
    server = VLLMServer(model_id=args.model_id, gpu=args.gpu)
    server.start()
    try:
        evaluations = [
            evaluate_prompt(
                server=server,
                examples=examples,
                prompt_config=prompt_config,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            for prompt_config in PROMPT_CONFIGS
        ]
    finally:
        # Stop the server even when one prompt evaluation raises an exception.
        server.stop()

    result = {
        "model_id": args.model_id,
        "dataset_path": str(args.dataset_path),
        "num_examples": len(examples),
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "n": 1,
            "seed": args.seed,
        },
        "evaluations": evaluations,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)

    for evaluation in evaluations:
        print(f"{evaluation['prompt_name']}: {evaluation['counts']}")
        print(
            "Manual review candidates: "
            f"format_1_answer_0={len(evaluation['manual_review']['format_1_answer_0'])}, "
            f"format_0_answer_0={len(evaluation['manual_review']['format_0_answer_0'])}"
        )


if __name__ == "__main__":
    main()
