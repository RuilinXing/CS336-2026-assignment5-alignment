"""Direct Preference Optimization on the HH data from the supplement."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.safety_rlhf import (
    HHPreferenceExample,
    compute_per_instance_dpo_loss,
    dpo_preference_margin,
    load_hh_preference_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, help="SFT checkpoint used for policy and reference.")
    parser.add_argument("--hh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-examples", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def validation_accuracy(
    model: torch.nn.Module,
    tokenizer,
    examples: list[HHPreferenceExample],
) -> float:
    if not examples:
        return float("nan")
    was_training = model.training
    model.eval()
    correct = 0
    for example in examples:
        margin = dpo_preference_margin(
            model,
            tokenizer,
            example.instruction,
            example.response_chosen,
            example.response_rejected,
        )
        correct += int(margin.item() > 0)
    if was_training:
        model.train()
    return correct / len(examples)


def save_checkpoint(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.validation_examples < 0:
        raise ValueError("validation_examples cannot be negative.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy_device = torch.device(args.policy_device)
    reference_device = torch.device(args.reference_device)
    if policy_device.type != "cuda" or reference_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPO training requires the requested CUDA devices.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
    policy = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(policy_device)
    reference = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(reference_device)
    policy.config.use_cache = False
    reference.config.use_cache = False
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    examples = load_hh_preference_dataset(args.hh_dir)
    random.shuffle(examples)
    validation_examples = examples[: args.validation_examples]
    training_examples = examples[args.validation_examples :]
    if not training_examples:
        raise ValueError("No preference examples remain after the validation split.")

    optimizer = RMSprop(policy.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_accuracy = float("-inf")
    for epoch in range(args.epochs):
        random.shuffle(training_examples)
        for index, example in enumerate(training_examples):
            window_start = (index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_size = min(args.gradient_accumulation_steps, len(training_examples) - window_start)
            loss = compute_per_instance_dpo_loss(
                policy,
                reference,
                tokenizer,
                args.beta,
                example.instruction,
                example.response_chosen,
                example.response_rejected,
            )
            (loss / window_size).backward()
            is_step = (index + 1) % args.gradient_accumulation_steps == 0 or index + 1 == len(training_examples)
            if not is_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(
                f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f}",
                flush=True,
            )
            if validation_examples and global_step % args.eval_every == 0:
                accuracy = validation_accuracy(policy, tokenizer, validation_examples)
                print(f"step={global_step} validation_accuracy={accuracy:.4f}", flush=True)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_checkpoint(policy, tokenizer, args.output_dir / "best")

    if validation_examples:
        accuracy = validation_accuracy(policy, tokenizer, validation_examples)
        print(f"final_validation_accuracy={accuracy:.4f}", flush=True)
        if accuracy > best_accuracy:
            save_checkpoint(policy, tokenizer, args.output_dir / "best")
    save_checkpoint(policy, tokenizer, args.output_dir / "final")


if __name__ == "__main__":
    main()
