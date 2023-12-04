from __future__ import annotations

import abc
import dataclasses as D
import enum
import functools
import itertools
import math
import typing as T
import warnings

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing_extensions import override

from unipercept.log import get_logger
from unipercept.utils.state import get_process_count, get_process_index

__all__ = ["TrainingSampler", "InferenceSampler", "SamplerFactory"]

_logger = get_logger(__name__)


@D.dataclass(slots=True)
class ProcessInfo:
    """
    Tuple representing the total number of distributed processes and the index of the active process.
    """

    count: int
    index: int

    def __post_init__(self):
        self.count = max(self.count, 1)


class BaseSampler(Sampler, metaclass=abc.ABCMeta):
    @staticmethod
    def get_dist_info(dist_num: int | None, dist_idx: int | None) -> ProcessInfo:
        """
        Returns the number of distributed processes (e.g. GPUs) and the index of the current process.
        If no value is provided, it is determined from the global state.

        In case parameters are provided, they must both be an integer.

        Parameters
        ----------
        dist_num
            The number of distributed processes.
        dist_idx
            The index of the current process.

        Returns
        -------
        dist_num
            The number of distributed processes.
        dist_idx
            The index of the current process.

        Raises
        ------
        ValueError
            If either ``dist_num`` or ``dist_idx`` is not an integer.
        """

        if not dist.is_available():
            raise RuntimeError("Distributed data sampler requires torch.distributed to be available.")

        if isinstance(dist_num, int) and isinstance(dist_idx, int):
            return ProcessInfo(count=dist_num, index=dist_idx)

        if dist_num is None and dist_idx is None:
            return ProcessInfo(count=get_process_count() or 1, index=get_process_index() or 0)

        raise ValueError(f"Both `dist_num` and `dist_idx` must be integers, but got {dist_num=} and {dist_idx=}.")

    _process_index: T.Final[int]
    _process_count: T.Final[int]

    def __init__(self, queue_size: int, *, process_index: int | None = None, process_count: int | None = None, epoch=0):
        assert queue_size > 0, f"Queue size must be positive, but got {queue_size=}."
        assert epoch >= 0, f"Epoch must be non-negative, but got {epoch=}."

        info = self.get_dist_info(process_index, process_count)

        self._process_count, self._process_index = info.count, info.index
        self._queue_size = queue_size
        self._epoch = epoch

        _logger.debug(f"Initialized sampler {self._process_index+1} of {self._process_count}")

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        _logger.debug(f"Sampler epoch set to {value}")
        self._epoch = value

    @property
    def process_index(self) -> int:
        return self._process_index

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def queue_size(self) -> int:
        return self._queue_size

    @property
    @abc.abstractmethod
    def indices(self) -> T.Iterator[int]:
        ...

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def total_count(self) -> int:
        ...

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self._epoch)

    @override
    def __iter__(self):
        yield from self.indices

    def __len__(self):
        raise NotImplementedError


class TrainingSampler(BaseSampler):
    def __init__(
        self, *args, shuffle=True, repeat_factor: float | int = 2, selected_round=0, selected_ratio=0.9, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._epoch = 0
        self._shuffle = shuffle
        self._repeat_factor = repeat_factor

        if not selected_ratio:
            selected_ratio = self._process_count
        if selected_round:
            assert selected_round > self.queue_size, f"{self.queue_size=} <= {selected_round=}."
            self._selected_count = int(math.floor(self.queue_size // selected_round * selected_round / selected_ratio))
        else:
            self._selected_count = int(math.ceil(self.queue_size / selected_ratio))

    @functools.cached_property
    @override
    def sample_count(self):
        return int(math.ceil(self.queue_size * self._repeat_factor / self.process_count))

    @functools.cached_property
    @override
    def total_count(self):
        return self.sample_count * self.process_count

    @property
    @override
    def indices(self):
        # Shuffle if needed
        if self._shuffle:
            idc = torch.randperm(self.queue_size, generator=self.generator)
        else:
            idc = torch.arange(start=0, end=self.queue_size)

        # Produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        rep = self._repeat_factor
        if isinstance(rep, float) and not rep.is_integer():
            rep_size = math.ceil(rep * self.queue_size)
            idc = idc[torch.tensor([int(i // rep) for i in range(rep_size)])]
        else:
            idc = torch.repeat_interleave(idc, repeats=int(rep), dim=0)

        idc = idc.tolist()
        # Add extra samples to make it evenly divisible
        pad_size = self.total_count - len(idc)
        if pad_size > 0:
            idc += idc[:pad_size]
        assert len(idc) == self.total_count

        # Subsample per process
        idc = idc[self.process_index : self.total_count : self.process_count]
        assert len(idc) == self.sample_count

        # Generate samples from the subsampled indices
        yield from iter(idc[: self._selected_count])

        self.epoch += 1

    @override
    def __len__(self):
        return min(self.sample_count, self._selected_count)


class InferenceSampler(BaseSampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    @staticmethod
    def create_indices(size: int, p_num: int, p_idx: int):
        shard_len = size // p_num
        shard_rem = size % p_num
        shard_sizes = [shard_len + int(r < shard_rem) for r in range(p_num)]

        i_start = sum(shard_sizes[:p_idx])
        i_end = min(sum(shard_sizes[: p_idx + 1]), size)

        return list(range(i_start, i_end))

    def __init__(self, *args, **kwargs):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        if "epoch" in kwargs:
            warnings.warn("Epoch argument is ignored in InferenceSampler.", UserWarning)
            del kwargs["epoch"]
        super().__init__(*args, **kwargs)

        self._indices = self.create_indices(self.queue_size, self.process_count, self.process_index)

    @property
    @override
    def epoch(self):
        raise ValueError("Epoch is not defined for InferenceSampler.")

    @property
    @override
    def indices(self):
        yield from iter(self._indices)

    @property
    @override
    def sample_count(self):
        return len(self._indices)

    @property
    @override
    def total_count(self):
        return self.queue_size

    @override
    def __len__(self):
        return self.sample_count


_P = T.ParamSpec("_P")


class SamplerType(enum.StrEnum):
    TRAINING = enum.auto()
    INFERENCE = enum.auto()


_SAMPLER_CLASS_MAP = {
    SamplerType.TRAINING: TrainingSampler,
    SamplerType.INFERENCE: InferenceSampler,
}


class SamplerFactory:
    __slots__ = ("_fn",)

    def __init__(self, sampler: SamplerType | str | T.Callable[T.Concatenate[int, _P], Sampler], **kwargs):
        if isinstance(sampler, (str, SamplerType)):
            init_fn = _SAMPLER_CLASS_MAP[SamplerType(sampler)]
        elif isinstance(sampler, type) and issubclass(sampler, Sampler):
            init_fn = sampler
        elif callable(sampler):
            _logger.warn(
                (
                    f"Could not explicitly determine whether `sampler` (type: {type(sampler)}) is a Sampler subclass or "
                    "name, assuming it is a callable that returns a subclass of `torch.utils.data.Sampler`. "
                    "This may lead to unexpected behavior. Please use `SamplerFactory` with a `SamplerType` or a "
                    "`torch.utils.data.Sampler` subclass instead."
                ),
                stacklevel=2,
            )
            init_fn = sampler

        self._fn = functools.partial(init_fn, **kwargs)

    def __call__(self, size: int) -> Sampler:
        return self._fn(size)
