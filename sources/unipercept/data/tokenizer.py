"""
Implements a simple BPE tokenizer from the GPT-2 model.
"""

from __future__ import annotations

import enum as E
import functools as F
import gzip
import html
import typing as T
import warnings
from importlib.abc import Traversable
from importlib.resources import files

import ftfy
import regex as re
import torch

from unipercept.file_io import Path
from unipercept.types import Pathable


class Tokenizer(T.Protocol):
    r"""
    Generic protocol for tokenizers.
    """

    SOT = "<|startoftext|>"
    EOT = "<|endoftext|>"

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@F.lru_cache
def bytes_to_unicode():
    r"""
    Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for decent
    coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables
    between utf-8 bytes and unicode strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=False))


def get_pairs(word):
    r"""
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class TokenizeTruncateMode(E.StrEnum):
    """
    Truncate modes for tokenization.
    """

    TRUNCATE = "truncate"
    """
    Truncate the input text to fit the maximum sequence length.
    """

    TRUNCATE_WARNING = "warning"
    """
    Print a warning if the input text is too long for the maximum sequence length,
    then truncate the input text.
    """

    RAISE = "raise"
    """
    Raise an error if the input text is too long for the maximum sequence length.
    """


class Tokenize:
    """
    Wraps a tokenizer to tokenize a list of strings.
    """

    def __init__(
        self,
        max_seq_len: int = 77,
        tokenizer: Tokenizer | None = None,
        truncate: TokenizeTruncateMode | str = TokenizeTruncateMode.TRUNCATE_WARNING,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or BytePairTokenizer()
        self.truncate = TokenizeTruncateMode(truncate)

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot = self.tokenizer.encode(self.tokenizer.SOT)
        eot = self.tokenizer.encode(self.tokenizer.EOT)
        all_tokens = [(sot + self.tokenizer.encode(text) + eot) for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                msg = f"Input {texts[i]} is too long for context length {self.max_seq_len}"
                match self.truncate:
                    case TokenizeTruncateMode.TRUNCATE:
                        tokens = tokens[: self.max_seq_len]
                    case TokenizeTruncateMode.TRUNCATE_WARNING:
                        warnings.warn(msg)
                        tokens = tokens[: self.max_seq_len]
                    case TokenizeTruncateMode.RAISE:
                        msg = f"Input {texts[i]} is too long for context length {self.max_seq_len}"
                        raise RuntimeError(msg)
                tokens[-len(eot) :] = eot
            result[i, : len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


class BytePairTokenizer(Tokenizer):
    """
    Tokenizer that uses Byte Pair Encoding (BPE) to encode and decode text.
    """

    DEFAULT_BPE = (
        files(__package__) if __package__ else Path(__file__).parent
    ).joinpath("tokenizer_bpe_16e6.txt.gz")

    def __init__(
        self,
        path: Pathable | Traversable = DEFAULT_BPE,
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        if not isinstance(path, (Traversable, Path)):
            path = Path(path)
        with path.open("rb") as fh:
            merges = gzip.open(fh).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend([self.SOT, self.EOT])

        self.encoder = dict(zip(vocab, range(len(vocab)), strict=False))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges)), strict=False))
        self.cache = {k: k for k in (self.SOT, self.EOT)}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:  # noqa: E722
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text
