"""
Commmon utilities for working with strings.
"""

import re

__all__ = ["convert_word_separator", "to_snake_case", "to_kebab_case"]

GROUP_BY_WORD = re.compile(r"(?<=[a-z\d])(?=[A-Z])|[^a-zA-Z\d\-/]")


def convert_word_separator(string: str, *, sep: str) -> str:
    """
    Converts a string to a different word separator.
    """
    return "".join(GROUP_BY_WORD.sub(" ", string).strip().replace(" ", sep))


def to_snake_case(string: str) -> str:
    """
    Converts a string to snake case.
    """
    return convert_word_separator(string, sep="_").lower()


def to_kebab_case(string: str) -> str:
    """
    Converts a string to kebab case.
    """
    return convert_word_separator(string, sep="-").lower()
