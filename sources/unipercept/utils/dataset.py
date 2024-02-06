"""
Implements a framework for working with datasets that have varying layouts and data types.
"""

from __future__ import annotations

import abc
import base64
import dataclasses as D
import functools
import pickle
import random
import types
import typing as T
import warnings

import numpy as np
import numpy.typing as NP
import torch
import torch.utils.data
from typing_extensions import override

from unipercept import file_io
from unipercept.utils.distributed import is_main_process, wait_for_sync
from unipercept.utils.frozendict import frozendict

__all__ = ["Dataset"]

_T_MFST = T.TypeVar("_T_MFST", bound=T.TypedDict)  # Manifest
_T_QITEM = T.TypeVar("_T_QITEM")  # Item in queue
_T_DITEM = T.TypeVar("_T_DITEM")  # Item in datapipe
_T_DINFO = T.TypeVar("_T_DINFO")  # Meta info about the dataset
_KEY_CREATE_INFO = "_create_info"


@T.dataclass_transform(field_specifiers=(D.Field, D.field), kw_only_default=True)
class DatasetMeta(abc.ABCMeta):
    @classmethod
    @override
    def __prepare__(cls, name, bases, **kwds):
        from unipercept.utils.pickle import as_picklable

        ns = {}

        # Set 'get_info' in the namespace of all classes of this metatype,
        info = kwds.pop("info", None)
        if info is None:
            # If not provided, then 'get_info' is not inherited but copied.
            info = next(
                (
                    getattr(base, _KEY_CREATE_INFO)
                    for base in bases
                    if hasattr(base, _KEY_CREATE_INFO)
                ),
                empty_info,
            )
        if callable(info):
            ns[_KEY_CREATE_INFO] = as_picklable(info)
        else:
            raise TypeError(f"info must be callable, got {type(info)}")

        return ns

    def __new__(metacls, name, bases, ns, *, extra_slots=(), **kwds):
        # Check whether slots is defined in the namespace
        has_slots = "__slots__" in ns
        if has_slots:
            ns["__slots__"] = tuple(set(ns["__slots__"]) | set(extra_slots))

        # Create new class
        bases = types.resolve_bases(bases)
        ds_cls = super().__new__(metacls, name, bases, ns, **kwds)

        # Convert to dataclass
        ds_cls = D.dataclass(slots=has_slots, weakref_slot=has_slots, kw_only=True)(ds_cls)  # type: ignore

        return ds_cls


def empty_info():
    return frozendict()


class Dataset(
    T.Generic[_T_MFST, _T_QITEM, _T_DITEM, _T_DINFO],
    metaclass=DatasetMeta,
    extra_slots=("__hash"),
):
    """
    Dataset class, which implements three main attributes: manifest, queue, and datapipe.

    Each attribute represents:
    - manifest: represents the results of discovering a dataset's files on the filesystem, e.g.
      a list of what files are in the dataset, and where they are located.
    - queue: represents an ordered and transformed version of the manifest that determines the
      way the data is loaded. For example, queue could be reordered per video sequence and made
      into pairs of 2 images from the manifest for training. The same dataset may define a different
      queue for validation, which could not have any pairing but instead be ordered by sequence and
      frame number such that the model can be evaluated on the entire sequence, and not just a pair.
    - datapipe: represents the output data stream of the dataclass, essentially transforming queue
      into data for the loader and model to use.

    Additionally, the dataset exposes an info function, which is a mutex that maps a dataset item ID, denoted by a tuple
    of (parent, key), to a dictionary of metadata about the item.
    This function is used to provide metadata to the datapipe, which is then used to transform the data into a format
    that can be used by the model.
    When the dataset consists of multiple sub-datasets, then the info function can be used to provide metadata about
    a specific sub-dataset by switching the parent argument.
    """

    use_manifest_cache: T.ClassVar[bool] = True

    @override
    def __init_subclass__(cls, use_manifest_cache: bool | None = None, **kwargs):
        super().__init_subclass__()

        if use_manifest_cache is not None:
            cls.use_manifest_cache = use_manifest_cache

    # -------- #
    # MANIFEST #
    # -------- #

    @abc.abstractmethod
    def _build_manifest(self) -> _T_MFST:
        """
        Builds the manifest attribute.

        This should represent a mapping of keys to values, where the keys are unique identifiers
        for each file in the dataset, and the values represent the available data for that identifier.

        For example, a dataset of images may have a manifest that looks like:
        {
            "image_0001": {
                "image_path": "/path/to/image_0001.png",
                "class": "cat",
                "segmentation_path": "/path/to/image_0001_segmentation.png",
                "size": (1920, 1080),
            },
            ...
        }

        The result is cached to disk, and the cache is shared across processes.
        """
        ...

    _manifest: _T_MFST | None = D.field(
        default=None, hash=False, repr=False, compare=False, init=False
    )

    @property
    def manifest(self) -> _T_MFST:
        """
        Manifest attribute: represents the results of discovering a dataset's files on the filesystem, e.g.
        a list of what files are in the dataset, and where they are located.
        """
        from unipercept.data.pipes import LazyPickleCache  # TODO: nasty dependency

        if self._manifest is None:
            if self.use_manifest_cache:
                # The manifest should be stored until the cache path provided by the environment
                file_name = base64.b64encode(
                    repr(self).encode(), "+-".encode()
                ).decode()
                path = file_io.get_local_path(
                    f"//cache/datasets/manifest_{file_name}.pth"
                )

                # Load the manifest from cache
                cache = LazyPickleCache(path)

                # Generate the manifest if it is not cached
                if not file_io.isfile(path) and is_main_process():
                    cache.data = self._build_manifest()

                # Wait while the manifest is being generated
                wait_for_sync()

                # Load from cache (also if main process)
                try:
                    mfst = cache.data  # type: ignore
                except Exception as e:
                    msg = f"Failed to load manifest from cache file: {path}"
                    raise RuntimeError(msg) from e

                mfst = self._manifest = cache.data  # type: ignore
            else:
                if is_main_process():
                    mfst = self._build_manifest()
                else:
                    mfst = None

                # Wait while the manifest is being generated
                wait_for_sync()

                # Send the manifest to all processes
                mfst = self._manifest = torch.distributed.broadcast(mfst, 0)
        else:
            mfst = self._manifest
        return T.cast(_T_MFST, mfst)

    # ----- #
    # QUEUE #
    # ----- #

    queue_fn: T.Callable[
        [_T_MFST], T.Mapping[str, _T_QITEM] | T.Iterable[tuple[str, _T_QITEM]]
    ] | None = D.field(default=None, repr=True, compare=False, kw_only=True)

    _queue: _Dataqueue[_T_QITEM] | None = D.field(
        default=None, hash=False, repr=False, compare=False, init=False
    )

    @property
    def queue(self) -> _Dataqueue[_T_QITEM]:
        """
        Queue attribute: represents an ordered and transformed version of the manifest that server
        a specific goal for loading. For example, queue could be reordered per video sequence and
        made into pairs of 2 images from the manifest for trainig.

        Returns a DatasetQueue object, which is a subclass of PyTorch dataset that implements
        a map-style dataset over unloaded data records. Can be indexed by key (string) and using
        the index number (int).
        """
        if self._queue is None:
            qmap = self.queue_fn(self.manifest)
            if isinstance(qmap, T.Mapping):
                qmap = dict(qmap)
            elif isinstance(qmap, T.Iterable):
                qmap = {k: v for k, v in qmap}
            else:
                raise TypeError(
                    f"Queue function must return a mapping or iterable, not {type(qmap)}"
                )

            if len(qmap) == 0:
                raise ValueError("Queue map must not be empty!")

            # Store for later
            self._queue = _Dataqueue(qmap)
        return self._queue

    # -------- #
    # DATAPIPE #
    # -------- #

    @classmethod
    @abc.abstractmethod
    def _load_data(cls, key: str, item: _T_QITEM, info: _T_DINFO) -> _T_DITEM:
        """
        Loads the data for a single item in the queue.
        """
        ...

    _datapipe: _Datapipe[_T_DITEM, _T_DINFO] | None = D.field(
        default=None, hash=False, repr=False, compare=False, init=False
    )

    @property
    def datapipe(self) -> _Datapipe[_T_DITEM, _T_DINFO]:
        """
        Datapipe attribute: represents the output dataset
        """
        if self._datapipe is None:
            self._datapipe = _Datapipe(
                self._load_data, queue=self.queue, info=self.info
            )
        return self._datapipe

    # --------------- #
    # INFO / METADATA #
    # --------------- #
    _create_info: T.ClassVar[T.Callable[[], _T_DINFO] | None]

    @classmethod
    @functools.lru_cache(maxsize=None)
    def read_info(cls) -> T.Any:
        """
        Info map attribute: represents the metadata for each item in the queue.
        """
        return cls._create_info()

    @property
    def info(self) -> _T_DINFO:
        """
        Info map attribute: represents the metadata for each item in the queue.
        """
        return self.read_info()


class _Dataqueue(torch.utils.data.Dataset[tuple[str, _T_QITEM]], T.Generic[_T_QITEM]):
    """
    A map-style dataset over unloaded data records. Records are pickled and concatenated
    into a large array. The array is then indexed using a mapping from keys to indices.
    """

    @staticmethod
    def _serialize(obj: object) -> NP.NDArray[np.uint8]:
        obj_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        obj_array = np.frombuffer(obj_bytes, dtype=np.uint8)

        return obj_array

    def __init__(self, queue_map: T.Mapping[str, _T_QITEM]):
        pkl_list = list(map(self._serialize, queue_map.values()))
        len_list = list(map(len, pkl_list))

        # Array of pickled objects and offset map
        self._array = torch.from_numpy(np.concatenate(pkl_list))
        self._addrs = torch.from_numpy(np.cumsum(len_list))

        # Translation objects
        self._key2idx = {str(key): idx for idx, key in enumerate(queue_map.keys())}
        self._idx2key = {idx: str(key) for idx, key in enumerate(queue_map.keys())}

    @override
    def __getitem__(self, key_or_idx: str | int) -> tuple[str, _T_QITEM]:
        # Find the appropriate index
        try:
            if isinstance(key_or_idx, str):
                idx = self._key2idx[key_or_idx]
                key = key_or_idx
            elif isinstance(key_or_idx, int):
                idx = key_or_idx
                key = self._idx2key[key_or_idx]
            else:
                raise TypeError(f"key must be str or int, not {type(key_or_idx)}")
        except KeyError as e:
            raise StopIteration from e

        # Select the bytes of the pickled object using the index and offset map
        start = 0 if idx == 0 else self._addrs[idx - 1]
        end = self._addrs[idx]
        obj_bytes = memoryview(self._array[start:end].numpy())

        # Unpickle the object and return with the key
        return key, pickle.loads(obj_bytes)

    def __iter__(self) -> T.Iterable[tuple[str, _T_QITEM]]:
        for key in self._key2idx.keys():
            key, item = self[key]
            yield key, item

    def __len__(self) -> int:
        return len(self._addrs)


class _Datapipe(
    torch.utils.data.Dataset[tuple[str, _T_DITEM]], T.Generic[_T_DITEM, _T_DINFO]
):
    """
    A map-style dataset over loaded data records.
    """

    def __init__(
        self,
        load_fn: T.Callable[[_T_QITEM, _T_DINFO], _T_DITEM],
        *,
        queue: _Dataqueue[_T_QITEM],
        info: _T_DINFO,
        retry: int = 0,
    ):
        self._retry = retry
        self._queue = queue
        self._load_fn = load_fn
        self._info = info

    def sample(self, k: int) -> T.Iterator[_T_DITEM]:
        rng = list(range(len(self)))
        idx = random.sample(rng, k)

        for i in idx:
            yield self[i]

    @override
    def __getitem__(self, key_or_idx: str | int) -> tuple[_T_DITEM]:
        key, item = self._queue[key_or_idx]
        return self._load_fn(key, item, self._info)

    def __iter__(self) -> T.Iterable[_T_DITEM]:
        queue = iter(self._queue)
        while True:
            for _ in range(self._retry + 1):
                try:
                    key, item = next(queue)
                    yield self._load_fn(key, item, self._info)
                    break
                except StopIteration:
                    return
                except Exception as e:
                    key = "UNKNOWN"
                    warnings.warn(f"Error loading item: {e}", stacklevel=2)
            else:
                raise RuntimeError(f"Failed to load item after {self._retry} retries")

    def __len__(self) -> int:
        return len(self._queue)
