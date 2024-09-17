from __future__ import annotations

import pytest
from unipercept.data.tokenizer import SimpleTokenizer, Tokenize


@pytest.fixture()
def tokenizer():
    return SimpleTokenizer()


def test_encode_decode(tokenizer: SimpleTokenizer):
    text_input = r"Hello, world!"
    text_output = r"hello , world !"

    # Check encode and decode consistency
    tokens = tokenizer.encode(text_input)
    assert text_output == tokenizer.decode(tokens).strip()

    # Change a token and check that it doesn't decode to the same text
    tokens[0] += 1
    assert text_output != tokenizer.decode(tokens).strip()

    # Check encode-decone-encode consistency
    assert tokens == tokenizer.encode(tokenizer.decode(tokens))


def test_tokenize(tokenizer):
    tokenize = Tokenize(tokenizer)
    text = "Hello, world!"
    tokens = tokenize(text)
    assert tokens.shape[0] == tokenize.max_seq_len
