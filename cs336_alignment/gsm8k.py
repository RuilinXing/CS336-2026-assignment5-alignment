"""读取 GSM8K 数据并生成作业使用的提示词。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GSM8KExample:
    """一个问题及其去除推理过程后的标准答案。"""

    question: str
    ground_truth: str


def load_gsm8k_examples(dataset_path: Path, max_examples: int | None = None) -> list[GSM8KExample]:
    """读取 JSONL，并将 ``####`` 后的内容作为奖励函数使用的最终答案。"""
    examples: list[GSM8KExample] = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            raw_example = json.loads(line)
            examples.append(
                GSM8KExample(
                    question=raw_example["question"],
                    ground_truth=raw_example["answer"].split("####")[-1].strip(),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def render_prompt(prompt_template: str, question: str) -> str:
    """把 GSM8K 问题填入提示词文件中的 ``{question}`` 占位符。"""
    return prompt_template.format(question=question)
