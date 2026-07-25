"""在 GSM8K 上运行标准 on-policy GRPO 训练。

示例：

    uv run python scripts/grpo.py --output-dir outputs/grpo_seed0 --seed 0

脚本默认采用作业讲义的 200 个 rollout step、256 条 response 的 rollout batch，
并使用两张 GPU：一张训练 Hugging Face 模型，另一张运行 vLLM 推理服务。
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.gsm8k import GSM8KExample, load_gsm8k_examples, render_prompt
from cs336_alignment.grpo import get_response_log_probs, grpo_train_step, tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AlgorithmSettings:
    """一个算法模式固定的奖励归一化和重要性重加权配置。"""

    baseline: str
    advantage_normalizer: str
    loss_normalization: str
    importance_reweighting_method: str
    uses_small_train_batch: bool = False


ALGORITHM_SETTINGS = {
    "standard": AlgorithmSettings("mean", "std", "sequence", "none"),
    "grpo_constant": AlgorithmSettings("mean", "std", "constant", "none"),
    "dr_grpo": AlgorithmSettings("mean", "none", "constant", "none"),
    "rft": AlgorithmSettings("none", "none", "constant", "none"),
    "maxrl": AlgorithmSettings("mean", "mean", "constant", "none"),
    "offpolicy_naive": AlgorithmSettings("mean", "std", "sequence", "none", True),
    "offpolicy_noclip": AlgorithmSettings("mean", "std", "sequence", "noclip", True),
    "offpolicy_grpo": AlgorithmSettings("mean", "std", "sequence", "grpo", True),
    "offpolicy_gspo": AlgorithmSettings("mean", "std", "sequence", "gspo", True),
    "offpolicy_cispo": AlgorithmSettings("mean", "std", "sequence", "cispo", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-dataset-path", type=Path, default=PROJECT_ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-dataset-path", type=Path, default=PROJECT_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument(
        "--rollout-format",
        choices=("r1_zero", "question_only"),
        default=None,
        help="Required when --prompt-path is a custom template, to select its matching reward and stop rule.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--num-train-examples", type=int, default=6400)
    parser.add_argument("--num-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--algorithm", choices=ALGORITHM_SETTINGS, default="standard")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--updates-per-rollout", type=int, default=None)
    parser.add_argument("--cliprange", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--rollout-log-every", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def resolve_training_values(args: argparse.Namespace) -> tuple[AlgorithmSettings, int, int, int, int | None, float | None]:
    """将算法名称展开为训练 step 实际需要的完整超参数。"""
    settings = ALGORITHM_SETTINGS[args.algorithm]
    default_train_batch_size = args.group_size if settings.uses_small_train_batch else args.rollout_batch_size
    train_batch_size = args.train_batch_size or default_train_batch_size
    default_accumulation_steps = 1 if settings.uses_small_train_batch else 32
    gradient_accumulation_steps = args.gradient_accumulation_steps or default_accumulation_steps
    default_updates = args.rollout_batch_size // train_batch_size if settings.uses_small_train_batch else 1
    updates_per_rollout = args.updates_per_rollout or default_updates
    normalization_constant = None
    if settings.loss_normalization == "constant":
        normalization_constant = train_batch_size * args.sampling_max_tokens
    default_cliprange = None
    if settings.importance_reweighting_method == "grpo":
        default_cliprange = 0.2
    elif settings.importance_reweighting_method in {"gspo", "cispo"}:
        default_cliprange = 3e-4 if settings.importance_reweighting_method == "gspo" else 0.2
    return (
        settings,
        train_batch_size,
        gradient_accumulation_steps,
        updates_per_rollout,
        normalization_constant,
        args.cliprange if args.cliprange is not None else default_cliprange,
    )


def validate_args(
    args: argparse.Namespace,
    settings: AlgorithmSettings,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    updates_per_rollout: int,
) -> None:
    """在启动昂贵的 GPU 作业前检查 batch 划分是否合理。"""
    if args.group_size <= 0:
        raise ValueError("group_size 必须为正数。")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size 必须能被 group_size 整除。")
    if train_batch_size % args.group_size != 0:
        raise ValueError("train_batch_size 必须能被 group_size 整除。")
    if train_batch_size > args.rollout_batch_size:
        raise ValueError("train_batch_size 不能大于 rollout_batch_size。")
    if not settings.uses_small_train_batch and train_batch_size != args.rollout_batch_size:
        raise ValueError("标准 on-policy GRPO 的 train_batch_size 必须等于 rollout_batch_size。")
    if settings.uses_small_train_batch:
        if args.rollout_batch_size % train_batch_size != 0:
            raise ValueError("off-policy 的 rollout_batch_size 必须能被 train_batch_size 整除。")
        if updates_per_rollout * train_batch_size != args.rollout_batch_size:
            raise ValueError("off-policy 的更新必须完整且恰好覆盖一个 rollout batch。")
    elif updates_per_rollout != 1:
        raise ValueError("标准 on-policy GRPO 每个 rollout batch 只能更新一次。")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size 必须能被 gradient_accumulation_steps 整除。")
    if args.num_rollout_steps <= 0 or args.eval_every <= 0 or updates_per_rollout <= 0:
        raise ValueError("num_rollout_steps、eval_every 和 updates_per_rollout 必须为正数。")


def set_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 的随机数，方便比较不同实验。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExperimentLogger:
    """同时写入本地 JSONL，并在用户指定时同步到 W&B。"""

    def __init__(
        self,
        output_dir: Path,
        config: dict[str, Any],
        wandb_project: str | None,
        wandb_run_name: str | None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (output_dir / "metrics.jsonl").open("w", encoding="utf-8")
        self.wandb_run = None
        if wandb_project is not None:
            import wandb

            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config,
            )

    def log(self, record: dict[str, Any]) -> None:
        """将同一条指标写入两个日志后端。"""
        self.metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.metrics_file.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(record, step=record["step"])

    def close(self) -> None:
        self.metrics_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def print_training_metrics(record: dict[str, Any]) -> None:
    """Print the assignment-required training or validation metrics."""
    if record["split"] == "train":
        metrics = [
            f"step={record['step']}",
            f"update={record['update_index']}",
            f"loss={record['loss']:.4f}",
            f"grad_norm={record['grad_norm']:.4f}",
            f"token_entropy={record['mean_token_entropy']:.4f}",
            f"train_reward={record['mean_reward']:.4f}",
            f"train_format_reward={record['mean_format_reward']:.4f}",
        ]
        if "clip_fraction" in record:
            metrics.append(f"clip_fraction={record['clip_fraction']:.4f}")
        print(" ".join(metrics), flush=True)
        return

    print(
        f"step={record['step']} val_reward={record['val_reward']:.4f} "
        f"val_format_reward={record['val_format_reward']:.4f} "
        f"val_average_response_length={record['val_average_response_length']:.1f}",
        flush=True,
    )


def resolve_rollout_configuration(
    prompt_path: Path,
    rollout_format: str | None = None,
) -> tuple[Callable[[str, str], dict[str, float]], list[str] | None]:
    """Choose the reward function and stop sequences required by a prompt template."""
    if rollout_format == "question_only":
        return question_only_reward_fn, None
    if rollout_format == "r1_zero":
        return r1_zero_reward_fn, ["</answer>"]

    prompt_path = prompt_path.resolve()
    question_only_path = (PROJECT_ROOT / "cs336_alignment/prompts/question_only.prompt").resolve()
    r1_zero_paths = {
        (PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt").resolve(),
        (PROJECT_ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt").resolve(),
    }
    if prompt_path == question_only_path:
        return question_only_reward_fn, None
    if prompt_path in r1_zero_paths:
        return r1_zero_reward_fn, ["</answer>"]
    raise ValueError("Custom --prompt-path requires --rollout-format.")


def build_sampling_params(
    args: argparse.Namespace,
    seed_offset: int = 0,
    stop_sequences: list[str] | None = None,
) -> dict[str, Any]:
    """集中保存生成参数，保证训练 rollout 与验证 rollout 使用同一配置。"""
    return {
        "temperature": args.sampling_temperature,
        "top_p": 1.0,
        "max_tokens": args.sampling_max_tokens,
        "n": 1,
        "seed": args.seed + seed_offset,
        "stop": stop_sequences,
        "include_stop_str_in_output": stop_sequences is not None,
    }


def cycle_batches(examples: list[GSM8KExample], batch_size: int) -> Iterable[list[GSM8KExample]]:
    """按顺序循环数据集，使训练步数可以超过单个 epoch。"""
    cursor = 0
    while True:
        batch = [examples[(cursor + offset) % len(examples)] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(examples)
        yield batch


def generate_responses(
    server: VLLMServer,
    prompts: list[str],
    sampling_params: dict[str, Any],
    batch_size: int,
) -> list[VLLMCompletion]:
    """调用 vLLM，并确认每个输入 prompt 都得到一个 completion。"""
    completions = server.generate_completions(prompts, sampling_params, batch_size=batch_size)
    expected_completions = len(prompts) * sampling_params.get("n", 1)
    if len(completions) != expected_completions:
        raise RuntimeError(f"vLLM 返回了 {len(completions)} 条结果，但期望 {expected_completions} 条。")
    return completions


def generate_grouped_responses(
    server: VLLMServer,
    prompts: list[str],
    group_size: int,
    sampling_params: dict[str, Any],
    batch_size: int,
) -> list[VLLMCompletion]:
    """Generate independent response groups for each unique prompt."""
    return generate_responses(
        server,
        prompts,
        {**sampling_params, "n": group_size},
        batch_size,
    )


@torch.no_grad()
def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    repeated_prompts: list[str],
    responses: list[str],
) -> torch.Tensor:
    """在更新前记录 rollout 的 token log-probability，供 off-policy 比率计算使用。"""
    device = next(model.parameters()).device
    tokenized = tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
    model.eval()
    scores = get_response_log_probs(
        model,
        tokenized["input_ids"].to(device),
        tokenized["labels"].to(device),
        return_token_entropy=False,
    )
    model.train()
    return scores["log_probs"].cpu()


def select_training_batch(
    repeated_prompts: list[str],
    responses: list[str],
    repeated_ground_truths: list[str],
    old_log_probs: torch.Tensor | None,
    update_index: int,
    train_batch_size: int,
) -> tuple[list[str], list[str], list[str], torch.Tensor | None]:
    """从一个 rollout batch 中选择当前更新使用的连续小 batch。"""
    start = (update_index * train_batch_size) % len(repeated_prompts)
    end = start + train_batch_size
    return (
        repeated_prompts[start:end],
        responses[start:end],
        repeated_ground_truths[start:end],
        old_log_probs[start:end] if old_log_probs is not None else None,
    )


def evaluate_policy(
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_template: str,
    sampling_params: dict[str, Any],
    batch_size: int,
    reward_fn: Callable[[str, str], dict[str, float]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """在验证集采样一次，并计算总奖励、格式奖励和平均 response 长度。"""
    prompts = [render_prompt(prompt_template, example.question) for example in examples]
    completions = generate_responses(server, prompts, sampling_params, batch_size)
    total_rewards = []
    format_rewards = []
    records = []
    for example, prompt, completion in zip(examples, prompts, completions, strict=True):
        reward = reward_fn(completion.text, example.ground_truth)
        total_rewards.append(reward["reward"])
        format_rewards.append(reward["format_reward"])
        records.append(
            {
                "question": example.question,
                "ground_truth": example.ground_truth,
                "prompt": prompt,
                "response": completion.text,
                "scores": reward,
            }
        )
    metrics = {
        "val_reward": sum(total_rewards) / len(total_rewards),
        "val_format_reward": sum(format_rewards) / len(format_rewards),
        "val_average_response_length": sum(len(item.token_ids) for item in completions) / len(completions),
    }
    return metrics, records


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    step: int,
) -> None:
    """保存可直接被 Hugging Face 和 vLLM 重新加载的模型与 tokenizer。"""
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def write_rollouts(output_dir: Path, step: int, records: list[dict[str, Any]]) -> None:
    """保存少量 rollout，方便人工检查模型的推理过程。"""
    rollout_dir = output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    with (rollout_dir / f"step_{step:04d}.json").open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    (
        settings,
        train_batch_size,
        gradient_accumulation_steps,
        updates_per_rollout,
        normalization_constant,
        cliprange,
    ) = resolve_training_values(args)
    validate_args(args, settings, train_batch_size, gradient_accumulation_steps, updates_per_rollout)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_gsm8k_examples(args.train_dataset_path, args.num_train_examples)
    val_examples = load_gsm8k_examples(args.val_dataset_path, args.num_val_examples)
    if not train_examples or not val_examples:
        raise ValueError("训练集和验证集都必须至少包含一个 GSM8K 样本。")
    random.Random(args.seed).shuffle(train_examples)
    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    reward_fn, stop_sequences = resolve_rollout_configuration(args.prompt_path, args.rollout_format)
    prompts_per_rollout = args.rollout_batch_size // args.group_size

    device = torch.device(f"cuda:{args.train_gpu}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    # 训练不会复用生成阶段的 KV cache，关闭它可以减少显存占用。
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    logger = ExperimentLogger(
        args.output_dir,
        {
            **vars(args),
            "resolved_train_batch_size": train_batch_size,
            "resolved_gradient_accumulation_steps": gradient_accumulation_steps,
            "resolved_updates_per_rollout": updates_per_rollout,
            "resolved_normalization_constant": normalization_constant,
            "resolved_cliprange": cliprange,
        },
        args.wandb_project,
        args.wandb_run_name,
    )
    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.vllm_gpu,
        port=args.vllm_port,
        seed=args.seed,
    )
    train_batches = cycle_batches(train_examples, prompts_per_rollout)

    server.start()
    try:
        # 首次同步会建立训练进程与 vLLM 进程之间的 NCCL 通信组。
        server.init_weight_sync(str(device))
        for step in range(1, args.num_rollout_steps + 1):
            prompt_examples = next(train_batches)
            prompts = [render_prompt(prompt_template, example.question) for example in prompt_examples]
            repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
            repeated_ground_truths = [
                example.ground_truth for example in prompt_examples for _ in range(args.group_size)
            ]

            # 每次生成前同步最新训练权重，才能保证这是 on-policy rollout。
            server.sync_policy_weights(model)
            completions = generate_grouped_responses(
                server,
                prompts,
                args.group_size,
                build_sampling_params(args, step, stop_sequences),
                args.rollout_batch_size,
            )
            responses = [completion.text for completion in completions]
            old_log_probs = None
            if settings.importance_reweighting_method != "none":
                old_log_probs = compute_old_log_probs(model, tokenizer, repeated_prompts, responses)

            for update_index in range(updates_per_rollout):
                (
                    train_prompts,
                    train_responses,
                    train_ground_truths,
                    train_old_log_probs,
                ) = select_training_batch(
                    repeated_prompts,
                    responses,
                    repeated_ground_truths,
                    old_log_probs,
                    update_index,
                    train_batch_size,
                )
                loss, train_metadata = grpo_train_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                    reward_fn=reward_fn,
                    repeated_prompts=train_prompts,
                    rollout_responses=train_responses,
                    repeated_ground_truths=train_ground_truths,
                    group_size=args.group_size,
                    baseline=settings.baseline,
                    advantage_normalizer=settings.advantage_normalizer,
                    importance_reweighting_method=settings.importance_reweighting_method,
                    old_log_probs=train_old_log_probs,
                    cliprange=cliprange,
                    loss_normalization=settings.loss_normalization,
                    normalization_constant=normalization_constant,
                )
                train_record = {
                    "step": step,
                    "update_index": update_index,
                    "algorithm": args.algorithm,
                    "split": "train",
                    "loss": loss.item(),
                    **train_metadata,
                }
                logger.log(train_record)
                print_training_metrics(train_record)

            if args.rollout_log_every and step % args.rollout_log_every == 0:
                write_rollouts(
                    args.output_dir,
                    step,
                    [
                        {
                            "question": example.question,
                            "ground_truth": example.ground_truth,
                            "prompt": prompt,
                            "response": completion.text,
                            "finish_reason": completion.finish_reason,
                            "scores": reward_fn(completion.text, example.ground_truth),
                        }
                        for example, prompt, completion in zip(
                            (example for example in prompt_examples for _ in range(args.group_size)),
                            repeated_prompts,
                            completions,
                            strict=True,
                        )
                    ],
                )

            if step % args.eval_every == 0:
                model.eval()
                server.sync_policy_weights(model)
                val_metrics, val_records = evaluate_policy(
                    server,
                    val_examples,
                    prompt_template,
                    build_sampling_params(args, 10_000 + step, stop_sequences),
                    args.eval_batch_size,
                    reward_fn,
                )
                validation_record = {"step": step, "split": "validation", **val_metrics}
                logger.log(validation_record)
                print_training_metrics(validation_record)
                write_rollouts(args.output_dir / "validation", step, val_records[:10])
                model.train()

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(model, tokenizer, args.output_dir, step)

        save_checkpoint(model, tokenizer, args.output_dir, args.num_rollout_steps)
    finally:
        server.stop()
        logger.close()


if __name__ == "__main__":
    main()
