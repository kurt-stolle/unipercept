"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import abc
import collections
import concurrent.futures
import functools
import gc
import itertools as I
import json
import os
import sys
import typing as T

import torch
import torch.multiprocessing as M
import torch.types
import typing_extensions as TX
from tensordict import (
    LazyStackedTensorDict,
    MemoryMappedTensor,
    PersistentTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import TensorDictFuture
from torch import Tensor

from unipercept import file_io
from unipercept.log import get_logger
from unipercept.state import (
    barrier,
    check_main_process,
    cpus_available,
    gather_tensordict,
    get_process_count,
    main_process_first,
    on_main_process,
)
from unipercept.utils.tensorclass import Tensorclass
from unipercept.utils.typings import Pathable

__all__ = ["ResultsWriter", "PersistentTensordictWriter", "MemmapTensordictWriter"]

_logger = get_logger(__name__)

_P = T.ParamSpec("_P")
_R = T.TypeVar("_R")


class ResultsWriter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, data: TensorDictBase) -> None:
        raise NotImplementedError("Abstract method `add` not implemented.")

    @abc.abstractmethod
    def flush(self) -> None:
        raise NotImplementedError("Abstract method `write` not implemented.")

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError("Abstract method `close` not implemented.")

    @property
    @abc.abstractmethod
    def tensordict(self) -> TensorDictBase:
        raise NotImplementedError("Abstract property `tensordict` not implemented.")


class LazyStackedMemmapTensorView:
    def __init__(self, path: str, key: str, index: tuple[int, int], dtype, shape):
        from tensordict.utils import _STRDTYPE2DTYPE

        self._path = path
        self._key = key
        self._index = index
        self._dtype = _STRDTYPE2DTYPE[dtype] if isinstance(dtype, str) else dtype
        self._shape = torch.Size(shape)

    def __len__(self) -> int:
        return self._index[1] - self._index[0]

    @TX.override
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getitem__(self, index: int | slice | tuple[int, int]) -> Tensor:
        if isinstance(index, int):
            return self._load_at(index)
        size = len(self)
        if isinstance(index, slice):
            start = index.start
            if start is None:
                start = 0
            if start < 0:
                start = size + start
                assert start >= 0
            stop = index.stop
            if stop is None:
                stop = size
            if stop < 0:
                stop = size + stop
                assert stop >= start
            step = index.step or 1
            loc = list(range(start, stop, step))
        else:
            loc = [i if i >= 0 else size + i for i in index]

        assert loc[0] > 0, f"{loc[0]} > 0, got: {loc} from index {index}"
        assert loc[1] > loc[0], f"{loc[1]} > {loc[0]}, got: {loc} from index {index}"
        assert loc[1] <= size, f"{loc[1]} > {size}, got: {loc} from index {index}"

        loc[0] += self._index[0]
        loc[1] += self._index[0]

        tensors: list[Tensor | None] = [None] * (loc[1] - loc[0])
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for i, t in pool.map(lambda i: (i, self._load_at(i)), range(*loc)):
                tensors[i] = t

        if any(t is None for t in tensors):
            raise RuntimeError("Some tensors were not loaded")
        return torch.stack(T.cast(list[torch.Tensor], tensors))

    def _load_at(self, i: int) -> Tensor:
        path = file_io.Path(self._path) / str(i) / f"{self._key}.memmap"
        tensor = (
            torch.from_file(str(path), dtype=self._dtype, size=self._shape.numel())
            .view(self._shape)
            .contiguous()
        )
        return tensor

    def to_tensor(self) -> Tensor:
        return self[:]


class LazyStackedMemmapTensorDict(TensorDictBase):
    _is_locked = True
    _is_memmap = True

    def __init__(
        self,
        path: str,
        index: tuple[int, int] | None = None,
        metadata: dict[str, T.Any] | None = None,
    ):
        self._path = path
        self._index = _find_memmap_indices(path) if index is None else index
        self._metadata = (
            metadata
            if metadata is not None
            else load_metadata(file_io.Path(path) / str(self._index[0]) / "meta.json")
        )

    @property
    @TX.override
    def batch_size(self) -> torch.Size:
        return torch.Size([self._index[1] - self._index[0]])

    def _index_tensordict(self, index, *args, **kwargs) -> T.Self:
        sub_td = self.__class__(self._path, index, self._metadata)
        return sub_td

    # ---------------------- #
    # Reading mapped tensors #
    # ---------------------- #

    @TX.override
    def _get_str(self, key, default) -> Tensor:
        if key not in self.keys():
            return default
        return LazyStackedMemmapTensorView(self._path, key, self._index).to_tensor()

    @TX.override
    def _get_at_str(self, key, idx, default) -> Tensor:
        meta = self._metadata.get(key, None)
        if meta is None:
            if default is None:
                raise KeyError(f"Key {key} not found in {self}")
            else:
                return default
        if isinstance(idx, int):
            idx_at = [idx, idx + 1]
        else:
            idx_at = idx
        tds = LazyStackedMemmapTensorView(
            self._path, key, idx_at, meta["dtype"], meta["shape"]
        )
        if isinstance(idx, int):
            return tds[0]
        return tds.to_tensor()

    @TX.override
    def _get_tuple(self, key, default) -> Tensor:
        key_str, *other = key
        if len(other) > 0:
            msg = f"Nested keys are not supported, got: {key}"
            raise NotImplementedError(msg)
        return self._get_str(key_str, default)

    @TX.override
    def _get_at_tuple(self, key, idx, default) -> Tensor:
        key_str, *other = key
        if len(other) > 0:
            msg = f"Nested keys are not supported, got: {key}"
            raise NotImplementedError(msg)
        return self._get_at_str(key_str, idx, default)

    # ----------------------- #
    # Functorch compatablilty #
    # ----------------------- #

    @TX.override
    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError("Cannot add batch dim to LazyConcatenatedMemmapTensorDict")

    @TX.override
    def _remove_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(
            "Cannot remove batch dim from LazyConcatenatedMemmapTensorDict"
        )

    # ------------ #
    # Dict methods #
    # ------------ #

    @TX.override
    def __setitem__(self, key: str, value: T.Any) -> T.NoReturn:
        raise NotImplementedError()

    @TX.override
    def keys(self) -> T.Iterator[str]:
        yield from (
            k
            for k, v in self._metadata.items()
            if isinstance(v, dict) and "dtype" in v and "shape" in v
        )

    @TX.override
    def values(self) -> T.Iterator[LazyStackedMemmapTensorView]:
        for _, value in self.items():
            yield value

    @TX.override
    def items(self) -> T.Iterator[T.Tuple[str, T.Any]]:
        for key in self.keys():
            yield key, LazyStackedMemmapTensorView(
                self._path,
                key,
                self._index,
                self._metadata[key]["dtype"],
                self._metadata[key]["shape"],
            )

    # ---------- #
    # Module API #
    # ---------- #

    @classmethod
    @TX.override
    def from_module(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} does not support from_module")

    @TX.override
    def to_module(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support to_module"
        )

    # ---------- #
    # State dict #
    # ---------- #

    @TX.override
    def state_dict(
        self, destination=None, prefix="", keep_vars=False, flatten=False
    ) -> OrderedDict[str, Any]:
        source = self
        out = collections.OrderedDict()
        out[prefix + "__path"] = source._path
        out[prefix + "__index"] = source._index
        if destination is not None:
            destination.update(out)
            return destination
        return out

    @TX.override
    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict=True,
        assign=False,
        from_flatten=False,
    ) -> T.Self:
        if assign:
            self._path = state_dict["__path"]
            return self
        else:
            return self.__class__(state_dict["__path"], state_dict["__index"])

    @TX.override
    def share_memory_(self):
        msg = f"{self.__class__.__name__} does not support share_memory_"
        raise NotImplementedError(msg)

    # ---------- #
    # Assignment #
    # ---------- #
    def __error_is_locked(self, *args, **kwargs) -> T.NoReturn:
        msg = f"{self.__class__.__name__} is locked"
        raise RuntimeError(msg)

    _set = __error_is_locked
    _set_str = __error_is_locked
    _set_tuple = __error_is_locked
    _set_at = __error_is_locked
    _set_at_str = __error_is_locked
    _set_at_tuple = __error_is_locked
    __setitem__ = __error_is_locked
    _set_non_tensor = __error_is_locked

    _update = __error_is_locked
    _update_at = __error_is_locked

    # ---------- #
    # Memmap API #
    # ---------- #
    def __error_is_memmaped(self, *args, **kwargs) -> T.NoReturn:
        msg = f"{self.__class__.__name__} is already memory-mapped"
        raise RuntimeError(msg)

    _memmap_ = __error_is_memmaped
    memmap_ = __error_is_memmaped
    memmap = __error_is_memmaped
    memmap_like = __error_is_memmaped

    @classmethod
    def load_memmap(cls, prefix: str | Path) -> T.Self:
        return cls(prefix)

    # ----------- #
    # Unsupported #
    # ----------- #

    def __eq__(self, *args, **kwargs):
        return NotImplementedError("Method __eq__ is not supported!")

    def __ne__(self, *args, **kwargs):
        return NotImplementedError("Method __ne__ is not supported!")

    def __or__(self, *args, **kwargs):
        return NotImplementedError("Method __or__ is not supported!")

    def __xor__(self, *args, **kwargs):
        return NotImplementedError("Method __xor__ is not supported!")

    def _apply_nest(self, *args, **kwargs):
        return NotImplementedError("Method _apply_nest is not supported!")

    def _change_batch_size(self, *args, **kwargs):
        return NotImplementedError("Method _change_batch_size is not supported!")

    def _check_device(self, *args, **kwargs):
        return NotImplementedError("Method _check_device is not supported!")

    def _check_is_shared(self, *args, **kwargs):
        return NotImplementedError("Method _check_is_shared is not supported!")

    def _clone(self, *args, **kwargs):
        return NotImplementedError("Method _clone is not supported!")

    def _convert_to_tensordict(self, *args, **kwargs):
        return NotImplementedError("Method _convert_to_tensordict is not supported!")

    def _erase_names(self, *args, **kwargs):
        return NotImplementedError("Method _erase_names is not supported!")

    def _exclude(self, *args, **kwargs):
        return NotImplementedError("Method _exclude is not supported!")

    def _has_names(self, *args, **kwargs):
        return NotImplementedError("Method _has_names is not supported!")

    def _load_memmap(self, *args, **kwargs):
        return NotImplementedError("Method _load_memmap is not supported!")

    def _permute(self, *args, **kwargs):
        return NotImplementedError("Method _permute is not supported!")

    def _rename_subtds(self, *args, **kwargs):
        return NotImplementedError("Method _rename_subtds is not supported!")

    def _select(self, *args, **kwargs):
        return NotImplementedError("Method _select is not supported!")

    def _squeeze(self, *args, **kwargs):
        return NotImplementedError("Method _squeeze is not supported!")

    def _stack_onto_(self, *args, **kwargs):
        return NotImplementedError("Method _stack_onto_ is not supported!")

    def _transpose(self, *args, **kwargs):
        return NotImplementedError("Method _transpose is not supported!")

    def _unbind(self, *args, **kwargs):
        return NotImplementedError("Method _unbind is not supported!")

    def _unsqueeze(self, *args, **kwargs):
        return NotImplementedError("Method _unsqueeze is not supported!")

    def _view(self, *args, **kwargs):
        return NotImplementedError("Method _view is not supported!")

    def all(self, *args, **kwargs):
        return NotImplementedError("Method all is not supported!")

    def any(self, *args, **kwargs):
        return NotImplementedError("Method any is not supported!")

    def contiguous(self, *args, **kwargs):
        return NotImplementedError("Method contiguous is not supported!")

    def del_(self, *args, **kwargs):
        return NotImplementedError("Method del_ is not supported!")

    def detach_(self, *args, **kwargs):
        return NotImplementedError("Method detach_ is not supported!")

    def device(self, *args, **kwargs):
        return NotImplementedError("Method device is not supported!")

    def entry_class(self, *args, **kwargs):
        return NotImplementedError("Method entry_class is not supported!")

    def expand(self, *args, **kwargs):
        return NotImplementedError("Method expand is not supported!")

    def is_contiguous(self, *args, **kwargs):
        return NotImplementedError("Method is_contiguous is not supported!")

    def masked_fill(self, *args, **kwargs):
        return NotImplementedError("Method masked_fill is not supported!")

    def masked_fill_(self, *args, **kwargs):
        return NotImplementedError("Method masked_fill_ is not supported!")

    def masked_select(self, *args, **kwargs):
        return NotImplementedError("Method masked_select is not supported!")

    def names(self, *args, **kwargs):
        return NotImplementedError("Method names is not supported!")

    def pin_memory(self, *args, **kwargs):
        return NotImplementedError("Method pin_memory is not supported!")

    def rename_key_(self, *args, **kwargs):
        return NotImplementedError("Method rename_key_ is not supported!")

    def reshape(self, *args, **kwargs):
        return NotImplementedError("Method reshape is not supported!")

    def split(self, *args, **kwargs):
        return NotImplementedError("Method split is not supported!")

    def to(self, *args, **kwargs):
        return NotImplementedError("Method to is not supported!")


class MemmapTensordictWriter(ResultsWriter):
    """
    Writes results to a MemmapTensor directory.

    See Also
    --------
    - TensorDict documentation: https://pytorch.org/tensordict/saving.html
    """

    def __init__(self, path: str, size: int, write_offset: int):
        """
        Parameters
        ----------
        path : str
            The path to the MemmapTensor directory.
        size : int
            The size of the first dimension of the results.
        """
        self._size: T.Final = size
        self._is_closed = False
        self._path = path
        self._pool = concurrent.futures.ThreadPoolExecutor(
            min(cpus_available(), M.cpu_count() // get_process_count(), 16)
        )
        self._futures = []

        self._td: TensorDict | None = None
        self._td_cursor = write_offset

    @TX.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self._path!r}, size={len(self)}, cursor={self._td_cursor})"

    @functools.cached_property
    def path(self) -> file_io.Path:
        p = file_io.Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __len__(self) -> int:
        return self._size

    @TX.override
    def flush(self):
        """
        Write the results to disk.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")
        concurrent.futures.wait(self._futures)
        self.close()

    @TX.override
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        for batched_item in data.cpu():
            futures = []
            batched_item._memmap_(
                prefix=self.path / str(self._td_cursor),
                inplace=True,
                like=False,
                futures=futures,
                executor=self._pool,
                copy_existing=False,
            )
            self._futures.extend(futures)

            self._td_cursor += 1

        self._futures = [f for f in self._futures if not f.done()]

    @TX.override
    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        self._pool.shutdown(wait=True)

    @property
    @TX.override
    def tensordict(self) -> TensorDictBase:
        if not self._is_closed:
            raise RuntimeError("ResultsWriter is not closed")
        if self._td is None:
            self._td = LazyStackedMemmapTensorDict(self._path, (0, self._size))
            # self._td = LazyStackedTensorDict._load_memmap(self.path, {"stack_dim": 0})
            # assert self._td.batch_size[0] == len(
            #    self
            # ), f"Expected batch size {len(self)}, got {self._td.batch_size[0]}"
        return self._td


class PersistentTensordictWriterProcess(M.Process):
    def __init__(
        self,
        path: Pathable,
        size: int,
        compression: T.Literal["lzf", "gzip"] | None,
        compression_opts: T.Any,
        queue: M.Queue,
    ):
        super().__init__()
        self.path = file_io.Path(path)
        self.size = size
        self.compression = compression
        self.compression_opts = compression_opts
        self.queue = queue
        self.cursor = 0

    @TX.override
    def run(self):
        _logger.debug("Starting H5 results writer process!")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = PersistentTensorDict(
            filename=self.path,
            batch_size=[self.size],
            mode="w",
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        try:
            while True:
                data = self.queue.get()
                if data is None:
                    _logger.debug("Received None, closing writer")
                    break
                writer[self.cursor : self.cursor + len(data)] = data
                self.cursor += len(data)
        finally:
            writer.close()


class PersistentTensordictWriter(ResultsWriter):
    """
    Writes results to a H5 file using PersistentTensorDict from multiple processes, uses a buffer to reduce the number of writes.
    """

    def __init__(
        self,
        path: Pathable,
        size: int,
        buffer_size: int = -1,
        compression: T.Literal["lzf", "gzip"] | None = "lzf",
        compression_opts: T.Any = None,
    ):
        """
        Parameters
        ----------
        path : str
            The path to the H5 file.
        size : int
            The size of the first dimension of the results.
        buffer_size : int, optional
            The size of the buffer, by default -1, which means no buffering.
        """
        self._size: T.Final = size
        self._is_closed = False
        self._path = file_io.Path(path)
        if self._path.is_dir():
            self._path = self._path / "results.h5"
        elif self._path.suffix not in (".h5", ".hdf5"):
            self._path = self._path.with_suffix(".h5")

        if check_main_process():
            self._queue = M.Queue(
                buffer_size if buffer_size > 0 else cpus_available() * 2
            )
            self._writer = PersistentTensordictWriterProcess(
                path, size, compression, compression_opts, queue=self._queue
            )
            self._writer.start()
        else:
            self._queue = None
            self._writer = None

        self._td: PersistentTensorDict | None = None
        self._td_factory = functools.partial(
            PersistentTensorDict,
            batch_size=[len(self)],
            mode="r",
            compression=compression,
            compression_opts=compression_opts,
        )

    def __del__(self):
        if not self._is_closed:
            _logger.warning("ResultsWriter was not closed, closing now", stacklevel=2)
        self.close()

    def __len__(self) -> int:
        return self._size

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        if self._queue is None:
            raise RuntimeError("ResultsWriter queue is None")
        self._queue.put(data.cpu())

    @TX.override
    def flush(self):
        """
        Write the results to disk.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        if check_main_process():
            assert self._queue is not None and self._writer is not None
            _logger.debug("Sending close signal to H5 writer")
            self._queue.put(None)
            _logger.debug("Waiting for H5 writer to finish")
            self._writer.join()
            self._writer.close()
            self._writer = None
        barrier()

        self.close()

    @TX.override
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        data = gather_tensordict(data)
        self._append(data)

    @TX.override
    def close(self):
        self._is_closed = True

        if check_main_process():
            if self._queue is not None:
                self._queue.close()
                self._queue = None
            if self._writer is not None:
                self._writer.terminate()
                self._writer.close()
                self._writer = None

        if self._td is not None:
            self._td.close()
            self._td = None

    @property
    @TX.override
    def tensordict(self) -> TensorDictBase:
        if not self._is_closed:
            raise RuntimeError("ResultsWriter is not closed")
        if self._td is None:
            self._td = self._td_factory(filename=self._path)
        return self._td


def _find_memmap_indices(path: Pathable) -> T.Tuple[int, int]:
    """
    Find the indices of the first and last memory-mapped file in a directory.

    Parameters
    ----------
    path : str
        The path to the directory.

    Returns
    -------
    T.Tuple[int, int]
        The first and last indices.
    """
    path = file_io.Path(path)
    indices = sorted(
        int(p.name)
        for p in path.iterdir()
        if int(p.name) > 0 and (p / "meta.json").is_file()
    )
    return indices[0], indices[-1]


def load_metadata(path: Pathable):
    with open(file_io.Path(path)) as json_metadata:
        metadata = json.load(json_metadata)
    return metadata


def _estimate_memory_footprint(o, handlers={}, verbose=False):
    """
    Calculates the approximate memory footprint an object and all of its contents.
    """
    dict_handler = lambda d: I.chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        collections.deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
        TensorDictBase: dict_handler,
        Tensorclass: dict_handler,
        T.Sequence: iter,
        T.Mapping: dict_handler,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
