"""生成或顺序执行 GRPO 多随机种子实验。

默认只打印命令，避免误启动耗时的双 GPU 训练；加入 ``--execute`` 才会逐条执行。

示例：

    uv run python scripts/run_grpo_experiments.py \
        --suite variants_on_policy --output-root outputs/variants
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    """一次训练运行所需的算法名称、随机种子与输出目录。"""

    algorithm: str
    seed: int
    output_dir: Path


SUITES = {
    "standard": ("standard",),
    "variants_on_policy": ("grpo_constant", "dr_grpo", "rft", "maxrl"),
    "off_policy": (
        "offpolicy_naive",
        "offpolicy_noclip",
        "offpolicy_grpo",
        "offpolicy_gspo",
        "offpolicy_cispo",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="standard")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "grpo_args",
        nargs=argparse.REMAINDER,
        help="传递给 scripts/grpo.py 的额外参数，必须放在 -- 后面。",
    )
    return parser.parse_args()


def selected_algorithms(suite: str) -> tuple[str, ...]:
    """将 ``all`` 展开成讲义中所有可比较的算法模式。"""
    if suite != "all":
        return SUITES[suite]
    return SUITES["standard"] + SUITES["variants_on_policy"] + SUITES["off_policy"]


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    """按算法、seed 的顺序生成彼此独立的输出目录。"""
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    return [
        Experiment(
            algorithm=algorithm,
            seed=seed,
            output_dir=args.output_root / algorithm / f"seed_{seed}",
        )
        for algorithm in selected_algorithms(args.suite)
        for seed in seeds
    ]


def build_command(args: argparse.Namespace, experiment: Experiment) -> list[str]:
    """生成单次训练命令；使用当前 Python 以复用 uv 已创建的虚拟环境。"""
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/grpo.py"),
        "--algorithm",
        experiment.algorithm,
        "--seed",
        str(experiment.seed),
        "--output-dir",
        str(experiment.output_dir),
        "--model-id",
        args.model_id,
        "--train-gpu",
        str(args.train_gpu),
        "--vllm-gpu",
        str(args.vllm_gpu),
    ]
    if args.wandb_project is not None:
        command.extend(["--wandb-project", args.wandb_project])
    return command + args.grpo_args


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args)
    commands = [build_command(args, experiment) for experiment in experiments]
    for command in commands:
        print(" ".join(command), flush=True)

    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
