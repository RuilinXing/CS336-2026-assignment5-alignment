from types import SimpleNamespace

import pytest
import torch

from .adapters import (
    run_aggregate_loss_across_microbatch,
    run_compute_group_normalized_rewards,
    run_compute_policy_gradient_loss,
    run_compute_rollout_rewards,
    run_get_response_log_probs,
    run_grpo_train_step,
    run_tokenize_prompt_and_output,
)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        del input_ids
        return SimpleNamespace(logits=self.fixed_logits)


class CountingUniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        self.batch_sizes.append(input_ids.shape[0])
        logits = self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def test_tokenization_aligns_response_labels_and_padding(tokenizer):
    result = run_tokenize_prompt_and_output(
        prompt_strs=["Hello", "Hello world"],
        output_strs=["world test", "test"],
        tokenizer=tokenizer,
    )

    assert result["input_ids"].tolist() == [[3, 4], [3, 4]]
    assert result["labels"].tolist() == [[4, 8], [4, 8]]
    assert result["response_mask"].tolist() == [[True, True], [False, True]]


def test_response_log_probs_and_entropy_match_manual_calculation():
    logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 0.0, -1.0]]])
    model = FixedLogitModel(logits)
    labels = torch.tensor([[2, 0]])
    input_ids = torch.tensor([[0, 0]])

    result = run_get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    expected_distribution = torch.log_softmax(logits, dim=-1)

    torch.testing.assert_close(
        result["log_probs"],
        expected_distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(
        result["token_entropy"],
        -(expected_distribution.exp() * expected_distribution).sum(dim=-1),
    )
    assert "token_entropy" not in run_get_response_log_probs(
        model, input_ids, labels, return_token_entropy=False
    )


def test_group_normalization_supports_all_requested_modes():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])

    std_advantages, _ = run_compute_group_normalized_rewards(
        rewards, group_size=2, baseline="mean", advantage_normalizer="std"
    )
    no_normalization, _ = run_compute_group_normalized_rewards(
        rewards, group_size=2, baseline="mean", advantage_normalizer="none"
    )
    mean_advantages, _ = run_compute_group_normalized_rewards(
        rewards, group_size=2, baseline="mean", advantage_normalizer="mean"
    )

    torch.testing.assert_close(std_advantages[:2], torch.tensor([2**-0.5, -(2**-0.5)]))
    torch.testing.assert_close(no_normalization, torch.tensor([0.5, -0.5, 0.0, 0.0]))
    torch.testing.assert_close(mean_advantages[:2], torch.tensor([1.0, -1.0]), rtol=1e-5, atol=1e-5)
    assert torch.isfinite(mean_advantages).all()


def test_rollout_rewards_preserve_order_and_report_component_means():
    calls: list[tuple[str, str]] = []

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        calls.append((response, ground_truth))
        reward = float(len(response))
        return {
            "reward": reward,
            "format_reward": reward + 1.0,
            "answer_reward": reward + 2.0,
        }

    rewards, metadata = run_compute_rollout_rewards(reward_fn, ["a", "bbb"], ["x", "y"])

    assert calls == [("a", "x"), ("bbb", "y")]
    torch.testing.assert_close(rewards, torch.tensor([1.0, 3.0]))
    assert metadata == {
        "mean_reward": 2.0,
        "mean_format_reward": 3.0,
        "mean_answer_reward": 4.0,
    }


def test_policy_gradient_clipping_respects_advantage_sign():
    policy_log_probs = torch.log(torch.tensor([[2.0, 0.5]]))
    old_log_probs = torch.zeros_like(policy_log_probs)

    positive_loss, _ = run_compute_policy_gradient_loss(
        torch.tensor([1.0]), policy_log_probs, "grpo", old_log_probs, cliprange=0.1
    )
    negative_loss, _ = run_compute_policy_gradient_loss(
        torch.tensor([-1.0]), policy_log_probs, "grpo", old_log_probs, cliprange=0.1
    )

    torch.testing.assert_close(positive_loss, torch.tensor([[-1.1, -0.5]]))
    torch.testing.assert_close(negative_loss, torch.tensor([[2.0, 0.9]]))


def test_cispo_keeps_a_clipped_gradient_for_large_importance_weights():
    policy_log_probs = torch.tensor([[torch.log(torch.tensor(2.0))]], requires_grad=True)
    old_log_probs = torch.zeros_like(policy_log_probs)

    loss, _ = run_compute_policy_gradient_loss(
        torch.tensor([1.0]), policy_log_probs, "cispo", old_log_probs, cliprange=0.1
    )
    loss.sum().backward()

    torch.testing.assert_close(policy_log_probs.grad, torch.tensor([[-1.1]]))


@pytest.mark.parametrize("method", ["grpo", "cispo"])
def test_clip_fraction_ignores_prompt_and_padding_tokens(method):
    policy_log_probs = torch.log(torch.tensor([[2.0, 1.0, 4.0]]))
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.tensor([[False, True, False]])

    _, metadata = run_compute_policy_gradient_loss(
        torch.tensor([1.0]),
        policy_log_probs,
        method,
        old_log_probs,
        cliprange=0.1,
        response_mask=response_mask,
    )

    torch.testing.assert_close(metadata["clip_fraction"], torch.tensor(0.0))


def test_gspo_uses_only_response_tokens_for_sequence_ratio():
    policy_log_probs = torch.log(torch.tensor([[2.0, 0.5, 4.0]]))
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.tensor([[True, False, True]])

    loss, _ = run_compute_policy_gradient_loss(
        torch.tensor([1.0]),
        policy_log_probs,
        "gspo",
        old_log_probs,
        cliprange=10.0,
        response_mask=response_mask,
    )

    expected_ratio = torch.sqrt(torch.tensor(8.0))
    torch.testing.assert_close(loss, -expected_ratio.expand_as(policy_log_probs))


def test_loss_aggregation_respects_mask_and_normalization():
    losses = torch.tensor([[2.0, 4.0, 100.0], [3.0, 9.0, 12.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    sequence_loss = run_aggregate_loss_across_microbatch(losses, mask, "sequence")
    constant_loss = run_aggregate_loss_across_microbatch(
        losses, mask, "constant", normalization_constant=4
    )

    torch.testing.assert_close(sequence_loss, torch.tensor(3.0))
    torch.testing.assert_close(constant_loss, torch.tensor(2.25))


def test_train_step_skips_zero_advantage_rollouts_and_clears_gradients(tokenizer):
    model = CountingUniformModel(len(tokenizer))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        del ground_truth
        reward = float(response == "world")
        return {"reward": reward, "format_reward": reward, "answer_reward": reward}

    initial_logits = model.logits.detach().clone()
    loss, _ = run_grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        reward_fn=reward_fn,
        repeated_prompts=["Hello"] * 4,
        rollout_responses=["world", "test", "test", "test"],
        repeated_ground_truths=["42"] * 4,
        group_size=2,
        baseline="none",
        advantage_normalizer="none",
        loss_normalization="constant",
        normalization_constant=4,
    )

    assert model.batch_sizes == [1]
    assert not torch.equal(model.logits.detach(), initial_logits)
    assert loss.ndim == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_step_clip_fraction_is_weighted_by_response_tokens(tokenizer):
    model = CountingUniformModel(len(tokenizer))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prompts = ["Hello", "Hello"]
    responses = ["world", "world test"]
    tokenized = run_tokenize_prompt_and_output(prompts, responses, tokenizer)
    uniform_log_prob = -torch.log(torch.tensor(float(len(tokenizer))))
    old_log_probs = torch.full_like(tokenized["input_ids"], uniform_log_prob, dtype=torch.float32)
    old_log_probs[0, tokenized["response_mask"][0]] -= torch.log(torch.tensor(2.0))

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        del response, ground_truth
        return {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}

    _, metadata = run_grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        reward_fn=reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=["42", "42"],
        group_size=1,
        baseline="none",
        advantage_normalizer="none",
        importance_reweighting_method="grpo",
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )

    assert metadata["clip_fraction"] == pytest.approx(1 / 3)


def test_off_policy_train_step_aligns_old_log_probs_to_selected_batch(tokenizer):
    model = CountingUniformModel(len(tokenizer))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    full_prompts = ["Hello", "Hello", "Hello"]
    full_responses = ["world test", "another test", "world test something something"]
    full_tokenized = run_tokenize_prompt_and_output(full_prompts, full_responses, tokenizer)
    selected_tokenized = run_tokenize_prompt_and_output(full_prompts[:2], full_responses[:2], tokenizer)
    assert full_tokenized["input_ids"].shape[1] > selected_tokenized["input_ids"].shape[1]
    old_log_probs = torch.zeros((2, full_tokenized["input_ids"].shape[1]))

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        del response, ground_truth
        return {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}

    loss, _ = run_grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        reward_fn=reward_fn,
        repeated_prompts=full_prompts[:2],
        rollout_responses=full_responses[:2],
        repeated_ground_truths=["42", "42"],
        group_size=1,
        baseline="none",
        advantage_normalizer="none",
        importance_reweighting_method="noclip",
        old_log_probs=old_log_probs,
    )

    assert torch.isfinite(loss)
