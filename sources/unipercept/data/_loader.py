"""Defines functions for creating dataloaders for training and validation, using the common dataset format."""

from __future__ import annotations

import dataclasses
import multiprocessing as M
import os
import typing as T

from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    Sampler,
    get_worker_info,
)
from typing_extensions import override

from unipercept.data.ops import Op, apply_dataset
from unipercept.log import get_logger
from unipercept.state import get_process_count
from unipercept.config import get_env

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["DataLoaderConfig", "DataLoaderFactory", "DatasetInterface"]

_logger = get_logger(__name__)

DEFAULT_NUM_WORKERS = max(1, get_env(int, "UNI_DATALOADER_WORKERS", "SLURM_CPUS_PER_GPU", default=M.cpu_count() // 4))


@dataclasses.dataclass(slots=True, frozen=True)
class DataLoaderConfig:
    drop_last: bool = False
    pin_memory: bool = True
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int | None = 4
    persistent_workers: bool | None = False


# V2
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

    dataset: "up.data.sets.PerceptionDataset"
    actions: T.Sequence[Op]
    sampler: "up.data.SamplerFactory"
    config: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    make_dataset_iterable: bool = dataclasses.field(
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
        loader_kwargs = {k: v for k, v in dataclasses.asdict(self.config).items() if v is not None}

        # Instantiate sampler
        queue_size = len(self.dataset.queue)
        sampler = self.sampler(queue_size)

        # Transform items in pipe
        datapipe = self.dataset.datapipe
        datapipe = apply_dataset(datapipe, self.actions)

        # Create a dataset inteface for the dataloader
        if self.make_dataset_iterable:
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

            _logger.debug("Transformed map-style dataset to iterable-style dataset: %s", str(interface))

            loader_kwargs["sampler"] = None
        else:
            interface = datapipe

            loader_kwargs["sampler"] = sampler

        # Loader
        _logger.debug(
            f"Creating dataloader from %d queued items in %d batches of %d",
            queue_size,
            len(interface),
            batch_size,
        )

        loader_kwargs["batch_size"] = batch_size
        loader_kwargs.setdefault("collate_fn", InputData.collate)
        loader_kwargs.setdefault("worker_init_fn", _worker_init_fn)
        return DataLoader(interface, **loader_kwargs)


# ------------------- #
# Dataset Preparation #
# ------------------- #


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


# ---------------- #
# Worker Functions #
# ---------------- #


# _T = T.TypeVar("_T", bound=InputData)

# def _collate_fn(batch: list[_T]) -> _T:
#     """Collate function that transforms a list of (id, data) tuples into a list of IDs and a stacked data object."""
#     import torch


#     # ids = [None] * len(batch)
#     # data_seq = [None] * len(batch)
#     # for i, (id, data) in enumerate(batch):
#     #     ids[i] = id
#     #     data_seq[i] = data

#     # assert all(id is not None for id in ids), "Some IDs are None!"
#     # assert all(data is not None for data in data_seq), "Some data are None!"

#     # return list(ids), torch.stack(data_seq)


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


# --------- #
# Utilities #
# --------- #


def _distribute_batch_size(total: int) -> int:
    """Given a total batch size, distribute it evenly across all GPUs."""
    world_size = get_process_count()
    if total == 0 or total % world_size != 0:
        raise ValueError(f"Total batch size ({total}) must be divisible by the number of gpus ({world_size}).")
    per_device = total // world_size

    return per_device
