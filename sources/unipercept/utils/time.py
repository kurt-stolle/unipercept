from __future__ import annotations

import typing as T
import time
from datetime import datetime
import contextlib
import statistics

from typing_extensions import override

__all__ = ["get_timestamp", "Profiler", "profile"]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class profile(contextlib.ContextDecorator):
    """
    Context manager that profiles a block of code and stores the elapsed time in a mapping.
    """
    def __init__(self, dest: T.MutableMapping, key: str):
        self.dest = dest
        self.key = key

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.dest[self.key] = elapsed_time

class ProfileAccumulator(T.MutableMapping):
    """
    Mapping that accumulates values can be used in combination with a Profiler, accumulating multiple values for the same key to gain insights over the distribution and evolution of the values over time.
    """

    def __init__(self):
        self.data = {}

    @override
    def __getitem__(self, key):
        return self.data[key]

    @override    
    def __setitem__(self, key, value):
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
    
    @property
    def means(self):
        return {k: sum(v)/len(v) for k, v in self.data.items()}
    
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
        return {k: statistics.stdev(v) for k, v in self.data.items()}
    
    @property
    def medians(self):
        return {k: statistics.median(v) for k, v in self.data.items()}
    
    @property
    def variances(self):
        return {k: statistics.variance(v) for k, v in self.data.items()}
    
    @property
    def counts(self):
        return {k: len(v) for k, v in self.data.items()}
    
    @property
    def items(self):
        return self.data.items
    
    @property
    def keys(self):
        return self.data.keys
    
    @property
    def values(self):
        return self.data.values
    
    @override 
    def __repr__(self):
        items: list[str] = []
        for k, m, s in zip(self.data.keys(), self.means.values(), self.stds.values()):
            items.append(f"{k}={(m/1000):.1f}±{(s/1000):.1f} ms")
        return f"{self.__class__.__name__}({', '.join(items)})"

    


