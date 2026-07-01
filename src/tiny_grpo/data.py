from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from tiny_grpo.tokenizer import TinyTokenizer


@dataclass(frozen=True)
class DigitExample:
    digit: int


class ParityDataset(Dataset[DigitExample]):
    def __init__(self, size: int = 10_000) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> DigitExample:
        return DigitExample(digit=index % 10)


class ParityCollator:
    def __init__(self, tokenizer: TinyTokenizer, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: list[DigitExample]) -> dict[str, torch.Tensor]:
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
    while True:
        yield from loader


def all_digit_prompts(
    tokenizer: TinyTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    digits = torch.arange(10, device=device)
    rows = [tokenizer.encode_prompt(int(digit.item())) for digit in digits]
    return torch.tensor(rows, dtype=torch.long, device=device), digits
