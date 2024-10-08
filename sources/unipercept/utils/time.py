from __future__ import annotations

import contextlib
import enum as E
import statistics
import time
import typing as T
from datetime import datetime
from typing import override

import pandas as pd

__all__ = ["get_timestamp", "ProfileAccumulator", "profile"]


class TimestampFormat(E.StrEnum):
    UNIX = E.auto()
    ISO = E.auto()
    LOCALE = E.auto()
    SHORT_YMD_HMS = E.auto()


def get_timestamp(*, format: str | TimestampFormat = TimestampFormat.ISO) -> str:
    """
    Returns a timestamp in the given format.
    """
    now = datetime.now()

    match format:
        case TimestampFormat.UNIX:
            return str(int(now.timestamp()))
        case TimestampFormat.ISO:
            return now.isoformat(timespec="seconds")
        case TimestampFormat.LOCALE:
            return now.strftime(r"%c")
        case TimestampFormat.SHORT_YMD_HMS:
            return now.strftime(r"%y%j%H%M%S")
        case _:
            raise ValueError(f"Invalid timestamp format: {format}")


class profile(contextlib.ContextDecorator):
    """
    Context manager that profiles a block of code and stores the elapsed time in a mapping.
    """

    def __init__(self, dest: T.MutableMapping | None, key: str):
        self.dest = dest
        self.key = key
        self.is_closed = dest is None

    def close(self):
        self.is_closed = True

    def __enter__(self):
        if not self.is_closed:
            self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_closed:
            return
        end_time = time.perf_counter_ns()
        elapsed_time = end_time - self.start_time
        self.dest[self.key] = elapsed_time


class ProfileAccumulator(T.MutableMapping):
    """
    Mapping that accumulates values can be used in combination with a Profiler, accumulating multiple values for the
    same key to gain insights over the distribution and evolution of the values over time.
    """

    _KEY_TIME: T.Final[str] = "seconds"
    _KEY_NAME: T.Final[str] = "name"

    def __init__(self):
        self.data = {}

    def run(
        self, warmup: int = 0, incomplete_ok: bool = True, enabled: bool = True
    ) -> T.Iterator[dict[str, int] | None]:
        """
        Returns an infinite iterator of dicts to which profiling data can be written.
        The first dict that is successfully returned is defines the keys that will be
        read.
        """
        if not enabled:
            while True:
                yield None
        skip = warmup
        while True:
            target = {}
            yield target
            if len(target.keys()) == 0:
                continue
            keys = list(target.keys())
            if len(self.data.keys()) > 0 and set(keys) != set(self.data.keys()):
                if not incomplete_ok:
                    msg = f"Keys must be consistent across profiling runs: {keys} != {self.data.keys()}"
                    raise ValueError(msg)
                else:
                    continue
            if skip > 0:
                skip -= 1
            else:
                for k, v in target.items():
                    self[k] = v
            del target

    @override
    def __getitem__(self, key):
        return self.data[key]

    @override
    def __setitem__(self, key, value: int):
        assert isinstance(value, int), "Value must be time in nanoseconds"
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    @override
    def __delitem__(self, key):
        del self.data[key]

    @override
    def __iter__(self):
        return iter(self.means)

    @override
    def __len__(self):
        return len(self.data)

    def reset(self):
        self.data.clear()

    @property
    def steps_recorded(self) -> int:
        if len(self.data) == 0:
            return 0
        steps = [len(v) for v in self.data.values()]
        return max(steps)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [(k, v * 1e-9) for k, iv in self.data.items() for _, v in enumerate(iv)],
            columns=[self._KEY_NAME, self._KEY_TIME],
        )

    def to_summary(self) -> pd.DataFrame:
        """
        Return summarizing statistics for every key in the accumulator.
        """
        df = self.to_dataframe()

        return df.groupby(self._KEY_NAME)[self._KEY_TIME].agg(
            ["count", "sum", "mean", "std", "min", "max", "median"]
        )

    @property
    def means(self):
        return {k: sum(v) / len(v) for k, v in self.data.items()}

    @property
    def sums(self):
        return {k: sum(v) for k, v in self.data.items()}

    @property
    def mins(self):
        return {k: min(v) for k, v in self.data.items()}

    @property
    def maxs(self):
        return {k: max(v) for k, v in self.data.items()}

    @property
    def stds(self):
        return {
            k: _compute_if_lengthy(v, statistics.stdev, 0, 1)
            for k, v in self.data.items()
        }

    @property
    def medians(self):
        return {
            k: _compute_if_lengthy(v, statistics.median, 0, 1)
            for k, v in self.data.items()
        }

    @property
    def variances(self):
        return {
            k: _compute_if_lengthy(v, statistics.variance, 0, 1)
            for k, v in self.data.items()
        }

    @property
    def counts(self):
        return {k: len(v) for k, v in self.data.items()}

    @property
    @override
    def items(self):
        return self.data.items

    @property
    @override
    def keys(self):
        return self.data.keys

    @property
    @override
    def values(self):
        return self.data.values

    @override
    def __repr__(self):
        items: list[str] = []
        for k, m, s in zip(
            self.data.keys(), self.means.values(), self.stds.values(), strict=False
        ):
            items.append(f"{k}={(m/1e6):.1f}±{(s/1e6):.1f} ms")
        return f"{self.__class__.__name__}({', '.join(items)})"


_L = T.TypeVar("_L", bound=list)
_R = T.TypeVar("_R", bound=T.Any)


def _compute_if_lengthy(
    value: _L, fn: T.Callable[[_L], _R], otherwise: _R, min_length: int = 0
) -> _R:
    if len(value) > min_length:
        return fn(value)
    return otherwise
