from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from tiny_grpo.tokenizer import TinyTokenizer


@dataclass(frozen=True)
class DigitExample:
    """One synthetic training example for the parity task."""

    digit: int


class ParityDataset(Dataset[DigitExample]):
    """A deterministic dataset that cycles through digits 0-9.

    The dataset does not store examples on disk. Index `i` maps to digit
    `i % 10`, so a shuffled DataLoader gives an endless-looking stream of small
    parity prompts while staying perfectly reproducible.
    """

    def __init__(self, size: int = 10_000) -> None:
        """Create a synthetic dataset of the requested length."""
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size

    def __len__(self) -> int:
        """Return the configured number of synthetic examples."""
        return self.size

    def __getitem__(self, index: int) -> DigitExample:
        """Return the digit example at `index`.

        The modulo pattern means the dataset contains a balanced mix of all ten
        digits when `size` is a multiple of 10.
        """
        return DigitExample(digit=index % 10)


class ParityCollator:
    """Turn `DigitExample` objects into model-ready tensors.

    PyTorch's DataLoader calls the collator once it has collected a Python list
    of dataset examples. The collator encodes prompts and moves both prompts and
    digit labels onto the selected device.
    """

    def __init__(self, tokenizer: TinyTokenizer, device: torch.device) -> None:
        """Store tokenizer and target device for future batches."""
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: list[DigitExample]) -> dict[str, torch.Tensor]:
        """Collate examples into a batch dictionary.

        Args:
            examples: A list of `DigitExample` objects from the dataset.

        Returns:
            A dictionary with:
            - `prompts`: LongTensor of shape `[batch_size, 2]`, where each row
              is `[<bos>, digit_token]`.
            - `digits`: LongTensor of shape `[batch_size]`, used by the reward
              function to decide whether `"odd"` or `"even"` is correct.
        """
        digits = torch.tensor([example.digit for example in examples], dtype=torch.long)
        prompts = [self.tokenizer.encode_prompt(int(digit.item())) for digit in digits]
        return {
            "prompts": torch.tensor(prompts, dtype=torch.long, device=self.device),
            "digits": digits.to(self.device),
        }


def make_parity_loader(
    tokenizer: TinyTokenizer,
    batch_size: int,
    device: torch.device,
    dataset_size: int = 10_000,
    seed: int = 0,
) -> DataLoader:
    """Build a shuffled DataLoader for the parity task.

    The loader yields dictionaries from `ParityCollator`. `drop_last=True` keeps
    every batch exactly `batch_size`, which makes the GRPO reshape from
    `[batch_size * group_size]` to `[batch_size, group_size]` simple.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ParityDataset(size=dataset_size),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=ParityCollator(tokenizer, device),
        generator=generator,
    )


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever by repeatedly iterating over `loader`.

    The training loop is step-based rather than epoch-based, so it asks for the
    next batch until `--steps` is reached.
    """
    while True:
        yield from loader


def all_digit_prompts(
    tokenizer: TinyTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one eval prompt for each digit 0-9.

    Returns:
        `(prompts, digits)` where prompts has shape `[10, 2]` and digits has
        shape `[10]`. Eval uses this to print `0->even`, `1->odd`, and so on.
    """
    digits = torch.arange(10, device=device)
    rows = [tokenizer.encode_prompt(int(digit.item())) for digit in digits]
    return torch.tensor(rows, dtype=torch.long, device=device), digits
