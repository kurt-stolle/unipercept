"""Defines functions for creating dataloaders for training and validation, using the common dataset format."""

from __future__ import annotations

import abc
import dataclasses
import dataclasses as D
import enum
import functools
import math
import multiprocessing as M
import typing as T
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    Sampler,
    get_worker_info,
)
from torch.utils.data.sampler import Sampler
from typing_extensions import override

from unipercept.config import get_env
from unipercept.data.ops import Op, apply_dataset
from unipercept.log import get_logger
from unipercept.state import get_process_count, get_process_index

if T.TYPE_CHECKING:
    from unipercept.data.sets import PerceptionDataset


__all__ = [
    "DataLoaderConfig",
    "DataLoaderFactory",
    "DatasetInterface",
    "TrainingSampler",
    "InferenceSampler",
    "SamplerFactory",
]

_logger = get_logger(__name__)

DEFAULT_NUM_WORKERS = max(
    1,
    get_env(
        int,
        "UNIPERCEPT_DATALOADER_WORKERS",
        default=get_env(int, "SLURM_CPUS_PER_GPU", default=M.cpu_count()) // 2,
    ),
)


@dataclasses.dataclass(slots=True, frozen=True)
class DataLoaderConfig:
    drop_last: bool = False
    pin_memory: bool = True
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int | None = 4
    persistent_workers: bool | None = False


##################
# Loader factory #
##################


@dataclasses.dataclass(slots=True, frozen=True)
class DataLoaderFactory:
    """
    Factory for creating dataloaders.

    Attributes
    ----------
    dataset
        The dataset to use.
    actions
        The actions to apply to the dataset (see: ``ops.py``).
    sampler
        The sampler to use.
    config
        The dataloader configuration to use.
    shard_sampler
        See: ``DatasetInterface``. Passing ``None`` uses that class's default.
    shard_chunk_size
        See: ``DatasetInterface``. Passing ``None`` uses that class's default.

    """

    dataset: PerceptionDataset
    actions: T.Sequence[Op]
    sampler: SamplerFactory
    config: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    iterable: bool = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "Whether to turn a MapDataset (i.e. a dataset that is not iterable) into an IterableDataset. "
                "See PyTorch DataLoader docs for more information."
            )
        },
    )

    def __call__(self, batch_size: int | None = None, /) -> DataLoader:
        from unipercept.data import SamplerFactory
        from unipercept.data.sets import PerceptionDataset
        from unipercept.model import InputData

        assert isinstance(self.dataset, PerceptionDataset), type(self.dataset)
        assert isinstance(self.sampler, SamplerFactory), type(self.sampler)
        assert isinstance(self.config, DataLoaderConfig), type(self.config)
        assert isinstance(self.actions, T.Sequence), type(self.actions)

        _logger.info("Wrapping dataset: %s", str(self.dataset))

        # Keyword arguments for the loader
        loader_kwargs = {
            k: v for k, v in dataclasses.asdict(self.config).items() if v is not None
        }

        # Instantiate sampler
        queue_size = len(self.dataset.queue)
        sampler = self.sampler(queue_size)

        # Transform items in pipe
        datapipe = self.dataset.datapipe
        datapipe = apply_dataset(datapipe, self.actions)

        # Create a dataset inteface for the dataloader
        if self.iterable:
            interface_kwargs = {}
            # TODO: find a way to pass these arguments in a way that is mutually exclusive `config.make_iterable`
            # if self.config.shard_sampler is not None:
            #     interface_kwargs["shard_sampler"] = self.shard_sampler
            # if self.config.shard_chunk_size is not None:
            #     interface_kwargs["shared_chunk_size"] = self.shard_chunk_size

            if isinstance(datapipe, IterableDataset):
                raise ValueError(
                    f"Dataset {self.dataset} is already an iterable dataset, cannot wrap it in another iterable dataset!"
                )

            interface = DatasetInterface(datapipe, sampler, **interface_kwargs)

            _logger.debug(
                "Transformed map-style dataset to iterable-style dataset: %s",
                str(interface),
            )

            loader_kwargs["sampler"] = None
        else:
            interface = datapipe

            loader_kwargs["sampler"] = sampler

        # Loader
        _logger.debug(
            "Creating dataloader from %d queued items in %d batches of %d",
            queue_size,
            len(interface),
            batch_size,
        )

        loader_kwargs["batch_size"] = batch_size
        loader_kwargs.setdefault("collate_fn", InputData.collate)
        loader_kwargs.setdefault("worker_init_fn", _worker_init_fn)
        return DataLoader(interface, **loader_kwargs)


#######################
# Dataset Preparation #
#######################


class DatasetInterface(IterableDataset):
    """
    Use a map-style dataset as an iterable dataset.

    Based on Detectron2 implementation: `detectron2.data.ToIterableDataset`.
    """

    @staticmethod
    def _roundrobin(*iterables):
        """Roundrobin('ABC', 'D', 'EF') --> A D E B F C."""
        from itertools import cycle, islice

        num_active = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    @staticmethod
    def _worker(iterable, *, chunk_size=1, strategy=_roundrobin):
        from itertools import islice

        # Shard the iterable if we're currently inside pytorch dataloader worker.
        worker_info = get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            # do nothing
            yield from iterable
        else:
            # worker0: 0, 1, ..., chunk_size-1, num_workers*chunk_size, num_workers*chunk_size+1, ...
            # worker1: chunk_size, chunk_size+1, ...
            # worker2: 2*chunk_size, 2*chunk_size+1, ...
            # ...
            yield from strategy(
                *[
                    islice(
                        iterable,
                        worker_info.id * chunk_size + chunk_i,
                        None,
                        worker_info.num_workers * chunk_size,
                    )
                    for chunk_i in range(chunk_size)
                ]
            )

    __slots__ = ("dataset", "sampler", "shard_sampler", "shard_chunk_size")

    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        shard_sampler: bool = True,
        shard_chunk_size: int = 1,
    ):
        """
        Parameters
        ----------
        dataset
            A map-style dataset.
        sampler
            A cheap iterable that produces indices to be applied on ``dataset``.
        shard_sampler
            Whether to shard the sampler based on the current pytorch data loader worker id.
            When an IterableDataset is forked by pytorch's DataLoader into multiple workers, it is responsible for
            sharding its data based on worker id so that workers don't produce identical data.
            Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
            and this argument should be set to True.
            But certain samplers may be already
            sharded, in that case this argument should be set to False.
        shard_chunk_size:
            When sharding the sampler, each worker will only produce 1/N of the ids
        """
        assert not isinstance(dataset, IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler
        self.shard_chunk_size = shard_chunk_size

    @override
    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = self._worker(self.sampler, chunk_size=self.shard_chunk_size)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore


####################
# Worker Functions #
####################


def _worker_init_fn(worker_id: int) -> None:
    """Worker init function that resets the random seed."""
    import os
    import random

    import numpy as np
    import torch

    seed = (torch.initial_seed() % 2**31) + worker_id

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


#############
# Utilities #
#############


def _distribute_batch_size(total: int) -> int:
    """Given a total batch size, distribute it evenly across all GPUs."""
    world_size = get_process_count()
    if total == 0 or total % world_size != 0:
        raise ValueError(
            f"Total batch size ({total}) must be divisible by the number of gpus ({world_size})."
        )
    per_device = total // world_size

    return per_device


@D.dataclass(slots=True)
class ProcessInfo:
    """
    Tuple representing the total number of distributed processes and the index of the active process.
    """

    count: int
    index: int

    def __post_init__(self):
        self.count = max(self.count, 1)


#################
# Sampler Types #
#################


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
            raise RuntimeError(
                "Distributed data sampler requires torch.distributed to be available."
            )

        if isinstance(dist_num, int) and isinstance(dist_idx, int):
            return ProcessInfo(count=dist_num, index=dist_idx)

        if dist_num is None and dist_idx is None:
            return ProcessInfo(
                count=get_process_count() or 1, index=get_process_index() or 0
            )

        raise ValueError(
            f"Both `dist_num` and `dist_idx` must be integers, but got {dist_num=} and {dist_idx=}."
        )

    _process_index: T.Final[int]
    _process_count: T.Final[int]

    def __init__(
        self,
        queue_size: int,
        *,
        process_index: int | None = None,
        process_count: int | None = None,
        epoch=0,
    ):
        assert queue_size > 0, f"Queue size must be positive, but got {queue_size=}."
        assert epoch >= 0, f"Epoch must be non-negative, but got {epoch=}."

        info = self.get_dist_info(process_index, process_count)

        self._process_count, self._process_index = info.count, info.index
        self._queue_size = queue_size
        self._epoch = epoch

        _logger.debug(
            f"Initialized sampler {self._process_index+1} of {self._process_count}"
        )

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
        self,
        *args,
        shuffle=True,
        repeat_factor: float | int = 2,
        selected_round=0,
        selected_ratio=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._epoch = 0
        self._shuffle = shuffle
        self._repeat_factor = repeat_factor

        if not selected_ratio:
            selected_ratio = self._process_count
        if selected_round:
            assert (
                selected_round > self.queue_size
            ), f"{self.queue_size=} <= {selected_round=}."
            self._selected_count = int(
                math.floor(
                    self.queue_size // selected_round * selected_round / selected_ratio
                )
            )
        else:
            self._selected_count = int(math.ceil(self.queue_size / selected_ratio))

    @functools.cached_property
    @override
    def sample_count(self):
        return int(
            math.ceil(self.queue_size * self._repeat_factor / self.process_count)
        )

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

        self._indices = self.create_indices(
            self.queue_size, self.process_count, self.process_index
        )

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

    def __init__(
        self,
        sampler: SamplerType | str | T.Callable[T.Concatenate[int, _P], Sampler],
        **kwargs,
    ):
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
