from __future__ import annotations

import re
from typing import Any


def full_name(mod: Any) -> str:
    """
    Returns the name of the class or instance.

    Parameters
    ----------
    mod
        Class or instance.

    Returns
    -------
    Name of the class or instance's class.
    """
    if isinstance(mod, type):
        return mod.__name__
    else:
        return mod.__class__.__name__


def short_name(mod: Any, num=4) -> str:
    """
    Abbreviates a module consistently across implementations using a
    simple common algorithm.

    Parameters
    ----------
    mod
        Class to abbreviate the name of.
    num
        The number of characters to abbreviate to.

    Returns
    -------
    The abbreviated module name.
    """
    name = full_name(mod)

    res = ["_"] * num
    res[: min(len(name), num)] = name[:num]

    def _assert_size():
        assert (
            len(res) == num
        ), f"Abbreviation of {name} yielded {str(res)} which is not of length {num}!"

    if len(name) > num:
        split = re.findall(r"[\dA-Z][^A-Z\d]*", name)

        if len(split) <= 1:
            no_vowels = [c for c in name[1:] if c not in "aeiou"]
            no_vowels.reverse()
            vowel_diff = num - len(no_vowels) - 1
            no_vowels = no_vowels[:vowel_diff]

            res.reverse()
            res[: min(len(no_vowels), len(res))] = no_vowels[: num - 1]
            res.reverse()

            _assert_size()
        elif len(split) >= num:
            first = [s[0] for s in split]
            res = first[: num - 1] + first[-1:]

            _assert_size()
        else:
            split = [s[0] + [c for c in s[1:] if c not in "aeiou"][0] for s in split]
            res = list(name)[:num]

            if len(split) == 2:
                size = num // 2
                res[size:] = split[1][:size]

                _assert_size()
            else:
                rem = len(split) - 1
                if rem > 0:
                    res[-rem:] = [s[0] for s in split][-rem:]

                _assert_size()

    return "".join(res).lower()
