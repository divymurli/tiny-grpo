from __future__ import annotations


class TinyTokenizer:
    def __init__(self) -> None:
        tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "odd",
            "even",
        ]
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_prompt(self, digit: int) -> list[int]:
        return [self.bos_id, self.token_to_id[str(digit)]]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for token_id in ids:
            token = self.id_to_token[int(token_id)]
            if token in {"<pad>", "<bos>"}:
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        return " ".join(tokens)

    def first_content_token(self, ids: list[int]) -> str | None:
        for token_id in ids:
            token = self.id_to_token[int(token_id)]
            if token in {"<pad>", "<bos>", "<eos>"}:
                continue
            return token
        return None
