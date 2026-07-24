"""Reusable components for the optional safety/RLHF supplement."""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


PROMPTS_DIR = Path(__file__).with_name("prompts_safety")
ALPACA_SFT_TEMPLATE = (PROMPTS_DIR / "alpaca_sft.prompt").read_text(encoding="utf-8")


def render_alpaca_sft(instruction: str, response: str = "") -> str:
    """Format an instruction-response pair with the supplement's Alpaca template."""
    return ALPACA_SFT_TEMPLATE.format(instruction=instruction, response=response)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """Extract the requested multiple-choice letter from a model response."""
    del mmlu_example
    match = re.search(
        r"\bthe\s+correct\s+answer\s+is\s*(?:option\s*)?\(?\s*([A-D])\s*\)?",
        model_output,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).upper()

    if re.fullmatch(r"\s*\(?\s*([A-D])\s*\)?\s*[.!]?\s*", model_output, re.IGNORECASE):
        return re.search(r"[A-D]", model_output, re.IGNORECASE).group(0).upper()
    return None


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def parse_gsm8k_response(model_output: str) -> str | None:
    """Return the final numeric value occurring in a GSM8K response."""
    matches = _NUMBER_PATTERN.findall(model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _open_jsonl(dataset_path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(dataset_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                yield json.loads(line)


class PackedSFTDataset(Dataset[dict[str, Tensor]]):
    """Constant-length next-token examples made from packed SFT documents."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be positive.")
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token.")

        documents = [
            render_alpaca_sft(example["prompt"], example["response"]).rstrip()
            for example in _open_jsonl(dataset_path)
        ]
        if shuffle:
            random.shuffle(documents)

        token_ids: list[int] = []
        for document in documents:
            token_ids.extend(tokenizer.encode(document))
            token_ids.append(tokenizer.eos_token_id)

        self.seq_length = seq_length
        self.token_ids = token_ids
        self.num_examples = max(0, (len(token_ids) - 1) // seq_length)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= self.num_examples:
            raise IndexError(index)
        start = index * self.seq_length
        end = start + self.seq_length
        return {
            "input_ids": torch.tensor(self.token_ids[start:end], dtype=torch.long),
            "labels": torch.tensor(self.token_ids[start + 1 : end + 1], dtype=torch.long),
        }


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a DataLoader whose iteration is exactly one dataset epoch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass(frozen=True)
class HHPreferenceExample:
    instruction: str
    response_chosen: str
    response_rejected: str
    source: str


_HH_CONVERSATION = re.compile(
    r"\A\s*Human:\s*(.*?)\s*Assistant:\s*(.*)\Z", re.DOTALL
)


def _split_single_turn_hh(conversation: str) -> tuple[str, str] | None:
    """Split a HH conversation, rejecting examples with a second human turn."""
    match = _HH_CONVERSATION.fullmatch(conversation)
    if match is None:
        return None
    instruction, response = match.groups()
    if re.search(r"\n\s*Human:\s*", instruction) or re.search(r"\n\s*Human:\s*", response):
        return None
    return instruction.strip(), response.strip()


def load_hh_preference_dataset(dataset_dir: str | Path) -> list[HHPreferenceExample]:
    """Load the four supplement HH collections and retain single-turn pairs."""
    filenames = (
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz",
    )
    examples: list[HHPreferenceExample] = []
    for filename in filenames:
        for record in _open_jsonl(Path(dataset_dir) / filename):
            chosen = _split_single_turn_hh(record["chosen"])
            rejected = _split_single_turn_hh(record["rejected"])
            if chosen is None or rejected is None or chosen[0] != rejected[0]:
                continue
            examples.append(
                HHPreferenceExample(
                    instruction=chosen[0],
                    response_chosen=chosen[1],
                    response_rejected=rejected[1],
                    source=filename.removesuffix(".jsonl.gz"),
                )
            )
    return examples


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("The language model must have parameters.") from error


def sequence_log_probability(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> Tensor:
    """Return the summed causal log-probability of a tokenized string."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        raise ValueError("Tokenized text must contain at least two tokens.")
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=_model_device(model)).unsqueeze(0)
    logits = model(input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum()


def _formatted_completion(instruction: str, response: str, tokenizer: PreTrainedTokenizerBase) -> str:
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define an EOS token string.")
    return render_alpaca_sft(instruction, response) + tokenizer.eos_token


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """Compute the Equation 3 DPO loss for one preference pair."""
    if beta <= 0:
        raise ValueError("beta must be positive.")

    chosen_text = _formatted_completion(prompt, response_chosen, tokenizer)
    rejected_text = _formatted_completion(prompt, response_rejected, tokenizer)
    policy_chosen = sequence_log_probability(lm, tokenizer, chosen_text)
    policy_rejected = sequence_log_probability(lm, tokenizer, rejected_text)
    with torch.no_grad():
        reference_chosen = sequence_log_probability(lm_ref, tokenizer, chosen_text)
        reference_rejected = sequence_log_probability(lm_ref, tokenizer, rejected_text)

    policy_device = _model_device(lm)
    preference_logit = beta * (
        (policy_chosen - reference_chosen.to(policy_device))
        - (policy_rejected - reference_rejected.to(policy_device))
    )
    return -F.logsigmoid(preference_logit)


def dpo_preference_margin(
    lm: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """Return the policy chosen-minus-rejected sequence log-probability."""
    return sequence_log_probability(
        lm, tokenizer, _formatted_completion(prompt, response_chosen, tokenizer)
    ) - sequence_log_probability(
        lm, tokenizer, _formatted_completion(prompt, response_rejected, tokenizer)
    )
