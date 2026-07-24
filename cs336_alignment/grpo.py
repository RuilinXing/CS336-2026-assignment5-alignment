from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """分别 token 化 prompt 与 response，并让掩码和 labels 对齐。"""
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    concatenated_ids = [
        prompt_token_ids + output_token_ids
        for prompt_token_ids, output_token_ids in zip(prompt_ids, output_ids, strict=True)
    ]
    max_length = max(len(token_ids) for token_ids in concatenated_ids)
    padded_input_ids = torch.full(
        (len(concatenated_ids), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    response_token_mask = torch.zeros(
        (len(concatenated_ids), max_length),
        dtype=torch.bool,
    )

    for index, (prompt_token_ids, token_ids) in enumerate(
        zip(prompt_ids, concatenated_ids, strict=True)
    ):
        length = len(token_ids)
        padded_input_ids[index, :length] = torch.tensor(token_ids, dtype=torch.long)
        response_token_mask[index, len(prompt_token_ids) : length] = True

    return {
        "input_ids": padded_input_ids[:, :-1],
        "labels": padded_input_ids[:, 1:],
        "response_mask": response_token_mask[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """计算标签 token 的 log-probability，以及可选的下一个 token 熵。"""
    logits = model(input_ids).logits
    log_distribution = torch.log_softmax(logits, dim=-1)
    result = {
        "log_probs": log_distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    }
    if return_token_entropy:
        result["token_entropy"] = -(log_distribution.exp() * log_distribution).sum(dim=-1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    """按 rollout 原有顺序打分，并汇总每种奖励分量。"""
    reward_components = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor(
        [components["reward"] for components in reward_components], dtype=torch.float32
    )
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": torch.tensor(
            [components["format_reward"] for components in reward_components], dtype=torch.float32
        ).mean().item(),
        "mean_answer_reward": torch.tensor(
            [components["answer_reward"] for components in reward_components], dtype=torch.float32
        ).mean().item(),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    """对每个 group 独立应用指定的 baseline 与归一化方式。"""
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if baseline == "mean":
        advantages = grouped_rewards - group_means
    else:
        advantages = grouped_rewards

    if advantage_normalizer == "std":
        advantages = advantages / (grouped_rewards.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_means + advantage_eps)

    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "mean_group_reward": group_means.mean().item(),
    }
    return advantages.reshape_as(raw_rewards), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo", "cispo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """计算尚未聚合的逐 token policy-gradient loss。"""
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    metadata: dict[str, Tensor] = {}

    def clip_fraction(clipped_positions: Tensor) -> Tensor:
        if response_mask is None:
            return clipped_positions.float().mean()
        mask = response_mask.to(dtype=clipped_positions.dtype)
        if mask.sum() == 0:
            return torch.zeros((), device=clipped_positions.device)
        return (clipped_positions.to(dtype=mask.dtype) * mask).sum() / mask.sum()

    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, metadata

    assert old_log_probs is not None
    log_ratios = policy_log_probs - old_log_probs
    token_ratios = log_ratios.exp()

    if importance_reweighting_method == "noclip":
        return -advantages * token_ratios, metadata

    assert cliprange is not None
    if importance_reweighting_method == "grpo":
        clipped_ratios = token_ratios.clamp(1.0 - cliprange, 1.0 + cliprange)
        objective = torch.minimum(advantages * token_ratios, advantages * clipped_ratios)
        metadata["clip_fraction"] = clip_fraction(torch.abs(token_ratios - 1.0) > cliprange)
        return -objective, metadata

    if importance_reweighting_method == "cispo":
        # CISPO 直接裁剪梯度系数；detach 保留该系数，但避免裁剪分支把梯度置零。
        gradient_coefficients = token_ratios.clamp(max=1.0 + cliprange).detach()
        metadata["clip_fraction"] = clip_fraction(token_ratios > 1.0 + cliprange)
        return -advantages * gradient_coefficients * policy_log_probs, metadata

    assert response_mask is not None
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)
    response_lengths = response_mask.sum(dim=1, keepdim=True)
    sequence_log_ratios = (log_ratios * response_mask).sum(dim=1, keepdim=True) / response_lengths
    sequence_ratios = sequence_log_ratios.exp()
    clipped_sequence_ratios = sequence_ratios.clamp(1.0 - cliprange, 1.0 + cliprange)
    objective = torch.minimum(
        advantages * sequence_ratios,
        advantages * clipped_sequence_ratios,
    )
    metadata["clip_fraction"] = (torch.abs(sequence_ratios - 1.0) > cliprange).float().mean()
    return -objective.expand_as(policy_log_probs), metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    """使用 response mask 聚合逐 token loss。"""
    masked_loss = per_token_policy_gradient_loss * mask.to(per_token_policy_gradient_loss.dtype)
    if loss_normalization == "sequence":
        sequence_losses = masked_loss.sum(dim=1) / mask.sum(dim=1)
        return sequence_losses.mean()

    assert normalization_constant is not None
    return masked_loss.sum() / normalization_constant


def _microbatch_loss(
    per_token_loss: Tensor,
    response_mask: Tensor,
    loss_normalization: Literal["sequence", "constant"],
    normalization_constant: int | None,
    original_batch_size: int,
) -> Tensor:
    if loss_normalization == "constant":
        return aggregate_loss_across_microbatch(
            per_token_loss,
            response_mask,
            loss_normalization="constant",
            normalization_constant=normalization_constant,
        )

    sequence_losses = (per_token_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1)
    return sequence_losses.sum() / original_batch_size


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo", "cispo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """使用预生成的 GRPO rollout 执行一次优化器更新。"""
    device = next(model.parameters()).device
    original_batch_size = len(repeated_prompts)
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    if old_log_probs is not None:
        sequence_length = tokenized["input_ids"].shape[1]
        if old_log_probs.ndim != 2 or old_log_probs.shape[0] != original_batch_size:
            raise ValueError("old_log_probs must have one row per rollout response.")
        if old_log_probs.shape[1] < sequence_length:
            raise ValueError("old_log_probs is shorter than the current tokenization.")
        old_log_probs = old_log_probs[:, :sequence_length]
    active_indices = torch.nonzero(advantages != 0, as_tuple=False).squeeze(1)

    optimizer.zero_grad(set_to_none=True)
    metadata: dict[str, Tensor | float] = {**reward_metadata, **advantage_metadata}
    if active_indices.numel() == 0:
        zero_loss = torch.zeros((), device=device)
        metadata["grad_norm"] = 0.0
        metadata["mean_token_entropy"] = 0.0
        return zero_loss, metadata

    input_ids = tokenized["input_ids"][active_indices].to(device)
    labels = tokenized["labels"][active_indices].to(device)
    response_mask = tokenized["response_mask"][active_indices].to(device)
    active_advantages = advantages[active_indices].to(device)
    active_old_log_probs = None
    if old_log_probs is not None:
        active_old_log_probs = old_log_probs[active_indices].to(device)

    microbatch_size = original_batch_size // gradient_accumulation_steps
    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_response_tokens = torch.zeros((), device=device)
    weighted_clip_fraction = torch.zeros((), device=device)
    clip_fraction_weight = torch.zeros((), device=device)

    for start in range(0, active_indices.numel(), microbatch_size):
        end = start + microbatch_size
        micro_input_ids = input_ids[start:end]
        micro_labels = labels[start:end]
        micro_mask = response_mask[start:end]
        micro_advantages = active_advantages[start:end]
        micro_old_log_probs = None
        if active_old_log_probs is not None:
            micro_old_log_probs = active_old_log_probs[start:end]

        response_scores = get_response_log_probs(
            model,
            micro_input_ids,
            micro_labels,
            return_token_entropy=True,
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            micro_advantages,
            response_scores["log_probs"],
            importance_reweighting_method,
            micro_old_log_probs,
            cliprange,
            micro_mask,
        )
        micro_loss = _microbatch_loss(
            per_token_loss,
            micro_mask,
            loss_normalization,
            normalization_constant,
            original_batch_size,
        )
        micro_loss.backward()
        total_loss = total_loss + micro_loss.detach()

        micro_response_tokens = micro_mask.sum()
        total_entropy = total_entropy + (response_scores["token_entropy"] * micro_mask).sum().detach()
        total_response_tokens = total_response_tokens + micro_response_tokens
        if "clip_fraction" in loss_metadata:
            if importance_reweighting_method in {"grpo", "cispo"}:
                micro_clip_weight = micro_response_tokens
            else:
                micro_clip_weight = torch.tensor(micro_input_ids.shape[0], device=device)
            weighted_clip_fraction = weighted_clip_fraction + (
                loss_metadata["clip_fraction"].detach() * micro_clip_weight
            )
            clip_fraction_weight = clip_fraction_weight + micro_clip_weight

    grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    metadata["grad_norm"] = grad_norm.item()
    metadata["mean_token_entropy"] = (total_entropy / total_response_tokens).item()
    if importance_reweighting_method in {"grpo", "gspo", "cispo"}:
        metadata["clip_fraction"] = (weighted_clip_fraction / clip_fraction_weight).item()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_loss, metadata
