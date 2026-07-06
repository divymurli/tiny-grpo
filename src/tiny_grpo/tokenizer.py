from __future__ import annotations


class TinyTokenizer:
    """A hand-written tokenizer for the toy parity environment.

    The vocabulary is intentionally tiny: special tokens, digit tokens `"0"` to
    `"9"`, and the two answer tokens `"odd"` and `"even"`. This lets the repo
    focus on GRPO mechanics without depending on an external tokenizer.
    """

    def __init__(self) -> None:
        """Build token/id lookup tables and expose common special token ids."""
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
        """Return the number of tokens the model can emit."""
        return len(self.token_to_id)

    def encode_prompt(self, digit: int) -> list[int]:
        """Encode one parity prompt as token ids.

        Args:
            digit: An integer from 0 to 9.

        Returns:
            A two-token prompt: `[<bos>, digit_token]`.
        """
        return [self.bos_id, self.token_to_id[str(digit)]]

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back into a readable completion string.

        `<pad>` and `<bos>` are skipped. Decoding stops at the first `<eos>`,
        mirroring how generated text is usually displayed.
        """
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
        """Return the first non-special token in a token sequence.

        The parity reward only cares whether the first content token is `"odd"`
        or `"even"`. If the sequence contains only special tokens, this returns
        `None`.
        """
        for token_id in ids:
            token = self.id_to_token[int(token_id)]
            if token in {"<pad>", "<bos>", "<eos>"}:
                continue
            return token
        return None
