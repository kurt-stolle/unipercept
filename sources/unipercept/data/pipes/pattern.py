from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import Iterator

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from typing_extensions import override

__all__ = ["PatternMatcher", "PatternFilter", "MatchMode"]


class MatchMode(Enum):
    """
    Pattern matcher modes.

    - ``error``: raise a ``ValueError`` if an item does not match the pattern.
    - ``warn``: raise a warning if an item does not match the pattern.
    - ``filter``: filter out items that do not match the pattern.
    - ``ignore``: ignore items that do not match the pattern, yielding None instead.
    """

    ERROR = "error"
    WARN = "warn"
    FILTER = "filter"
    IGNORE = "ignore"


@functional_datapipe("match_pattern_by_unicore")
class PatternMatcher(IterDataPipe):
    """
    Iterable pipe that performs pattern matching.
    """

    def __init__(self, source: IterDataPipe[str], pattern: str | re.Pattern, mode: str | MatchMode) -> None:
        self.source = source
        self.pattern = re.compile(pattern)
        self.mode = MatchMode(mode)

    @override
    def __iter__(self) -> Iterator[tuple[re.Match | None, str]]:
        for item in self.source:
            m = self.pattern.search(item)
            if m is None:
                match self.mode:
                    case MatchMode.ERROR:
                        raise ValueError(f"Item '{item}' does not match pattern: {self.pattern}")
                    case MatchMode.WARN:
                        warnings.warn(f"Item '{item}' does not match pattern: {self.pattern}", stacklevel=2)
                    case MatchMode.FILTER:
                        continue
                    case MatchMode.IGNORE:
                        pass
            yield (m, item)


@functional_datapipe("filter_pattern_by_unicore")
class PatternFilter(IterDataPipe):
    """
    Iterable pipe that performs pattern matching and filtering.
    """

    def __init__(self, source: IterDataPipe[str], pattern: str | re.Pattern, reverse=False) -> None:
        self.source = source
        self.pattern = re.compile(pattern)
        self.reverse = reverse

    @override
    def __iter__(self) -> Iterator[str]:
        for item in self.source:
            m = self.pattern.search(item)
            if (m is not None) ^ self.reverse:  # xor
                yield item
