"""Supervised instruction fine-tuning for the safety/RLHF supplement."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.safety_rlhf import PackedSFTDataset, iterate_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dev-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def language_modeling_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in data_loader:
        batch_loss = language_modeling_loss(model, batch, device)
        batch_size = batch["input_ids"].shape[0]
        total_loss += batch_loss.item() * batch_size
        total_examples += batch_size
    if was_training:
        model.train()
    return total_loss / total_examples if total_examples else float("nan")


def save_checkpoint(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    if args.micro_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch sizes and gradient accumulation steps must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": torch.bfloat16} if device.type == "cuda" else {}
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(device)
    model.config.use_cache = False

    train_dataset = PackedSFTDataset(tokenizer, args.train_path, args.seq_length, shuffle=True)
    train_loader = iterate_batches(train_dataset, args.micro_batch_size, shuffle=True)
    if not len(train_loader):
        raise ValueError("The packed training dataset is empty.")
    dev_loader = None
    if args.dev_path is not None:
        dev_dataset = PackedSFTDataset(tokenizer, args.dev_path, args.seq_length, shuffle=False)
        dev_loader = iterate_batches(dev_dataset, args.micro_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_multiplier)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_dev_loss = float("inf")
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            window_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_size = min(args.gradient_accumulation_steps, len(train_loader) - window_start)
            loss = language_modeling_loss(model, batch, device)
            (loss / window_size).backward()
            is_step = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if not is_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(
                f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
            if dev_loader is not None and global_step % args.eval_every == 0:
                dev_loss = evaluate(model, dev_loader, device)
                print(f"step={global_step} dev_loss={dev_loss:.4f}", flush=True)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    save_checkpoint(model, tokenizer, args.output_dir / "best")
            if global_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.output_dir / f"checkpoint-{global_step}")

    if dev_loader is not None:
        dev_loss = evaluate(model, dev_loader, device)
        print(f"final_dev_loss={dev_loss:.4f}", flush=True)
        if dev_loss < best_dev_loss:
            save_checkpoint(model, tokenizer, args.output_dir / "best")
    save_checkpoint(model, tokenizer, args.output_dir / "final")


if __name__ == "__main__":
    main()
