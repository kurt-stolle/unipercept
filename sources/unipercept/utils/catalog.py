"""
Implements a simple data manager for registering datasets and their info functions.
"""

from __future__ import annotations

import importlib.metadata
import re
import typing as T

import typing_extensions as TX

from unipercept.utils.dataset import Dataset
from unipercept.utils.registry import Registry

__all__ = ["DataManager", "DataManagerWithMetadataGroup"]


_DEFAULT_ID_PATTERN: T.Final[re.Pattern] = re.compile(r"^[a-z\d\-]+$")
_DEFAULT_ID_SEPARATOR: T.Final[str] = "/"

_D_co = T.TypeVar("_D_co", covariant=True)
_I_co = T.TypeVar("_I_co", covariant=True)
_P = T.ParamSpec("_P")


class DataManager(T.Generic[_D_co, _I_co]):
    """
    Data manager for registering datasets and their info functions.
    """

    __slots__ = (
        "name",
        "variant_separator",
        "id_pattern",
        "_info_registry",
        "_data_registry",
    )

    def __init__(
        self,
        *,
        id_pattern: re.Pattern[str] = _DEFAULT_ID_PATTERN,
        variant_separator: str = _DEFAULT_ID_SEPARATOR,
    ):
        """
        Parameters
        ----------
        id_pattern : re.Pattern
            The pattern to use for validating dataset IDs.
        variant_separator : str
            The separator to use for separating dataset IDs from variant IDs.
        """
        self.variant_separator: T.Final[str] = variant_separator
        self.id_pattern: T.Final[re.Pattern[str]] = id_pattern

        self._info_registry = T.cast(
            Registry[T.Callable[..., _I_co], str], Registry(self.parse_key)
        )
        self._data_registry = T.cast(
            Registry[type[_D_co], str], Registry(self.parse_key)
        )

    def parse_key(self, key: str | type[_D_co], *, check_valid: bool = True) -> str:
        """
        Convert a string or class to a canonical ID.

        Parameters
        ----------
        other : Union[str, type]
            The string or class to convert.

        Returns
        -------
        str
            The canonical ID.
        """
        id_ = key if isinstance(key, str) else key.__name__.replace("Dataset", "")
        id_ = id_.lower()
        if check_valid and not self.id_pattern.match(id_):
            raise ValueError(f"{id_} ({key}) does not match {self.id_pattern.pattern}")

        return id_

    def split_query(self, query: str) -> tuple[str, frozenset[str]]:
        """
        Split a query into a dataset ID and a variant ID.
        """
        if self.variant_separator not in query:
            return query, []
        else:
            key, *variant = query.split(self.variant_separator, maxsplit=1)
            return key, variant

    def __ior__(self, __other: DataManager, /) -> T.Self:
        """
        Merge the data and info registries of this manager with another.
        The other manager takes precedence in case of conflicts.
        """
        self._data_registry |= __other._data_registry
        self._info_registry |= __other._info_registry

        return self

    def __or__(self, __other: DataManager, /) -> T.Self:
        from copy import copy

        obj = copy(self)
        obj |= __other

        return obj

    def fork(self) -> DataManager:
        """
        Return a copy of this data manager.
        """
        return DataManager() | self

    # -------- #
    # DATASETS #
    # -------- #

    def register_dataset(
        self, id: str | None = None, *, info: T.Optional[T.Callable[..., _I_co]] = None
    ) -> T.Callable[[type[Dataset]], type[Dataset]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : Optional[str]
            The ID to register the dataset with. If None, the dataset class name will be used (flattened and converted
            to snake_case).
        """

        def wrapped(ds: type[_D_co]) -> type[_D_co]:
            key = id or self.parse_key(ds)
            self._data_registry[key] = ds
            if info is None:
                raise ValueError(
                    f"Dataset {key} has no info function and no info function was provided."
                )
            if callable(info):
                self._info_registry[key] = info
            else:
                raise TypeError(f"Invalid info function: {info}")

            return ds

        return wrapped

    def get_dataset(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        return self._data_registry[query]

    def list_datasets(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return frozenset(self._data_registry.keys())

    # ---- #
    # Info #
    # ---- #

    def register_info(
        self,
        key: str,
        /,
    ) -> T.Callable[[T.Callable[_P, _I_co]], T.Callable[_P, _I_co]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : Optional[str]
            The ID to register the dataset with. If None, the dataset class name will be canonicalized using
            ``canonicalize_id``.
        """

        def wrapped(info: T.Callable[_P, _I_co]) -> T.Callable[_P, _I_co]:
            self._info_registry[key] = info

            return info

        return wrapped

    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)

    def get_info_at(self, query: str, key: str) -> T.Any:
        """
        Return the info for the given dataset ID.
        """
        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)[key]  # type: ignore

    def list_info(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return frozenset(self._info_registry.keys())


class DataManagerWithMetadataGroup(DataManager[_D_co, _I_co], T.Generic[_D_co, _I_co]):
    """
    Variant of :class:`DataManager` that reads registered items from the metadata
    registered through ``importlib.metadata``.

    Notes
    -----
    This comes with the restriction that each registered ID can only reference both
    a dataset and an info function.
    """

    __slots__ = "group"

    def __init__(
        self,
        *,
        group: str,
        **kwargs,
    ):
        """
        Parameters
        ----------
        group : str
            The metadata group to read from.
        **kwargs
            See :class:`DataManager`.
        """
        super().__init__(**kwargs)

        self.group = group

    def list_entrypoints(self) -> frozenset[str]:
        """
        Returns a list of all registered keys from ``importlib.metadata`` with
        ``self.group``.
        """
        return frozenset(importlib.metadata.entry_points(group=self.group).names)

    def get_entrypoint(self, query: str) -> type[_D_co]:
        """
        Return the entrypoint for the given dataset ID.
        """
        return importlib.metadata.entry_points(group=self.group)[query].load()

    @TX.override
    def get_dataset(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        try:
            return self._data_registry[query]
        except KeyError:
            return self.get_entrypoint(query)

    @TX.override
    def list_datasets(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        reg_ds = set(super().list_datasets())
        meta_ds = set(self.list_entrypoints())

        return frozenset(reg_ds | meta_ds)

    def _maybe_load_entrypoint(self, query: str, *, raises: bool = True) -> None:
        _id, _ = self.split_query(query)
        if _id in super().list_info():
            return
        if _id not in self.list_entrypoints():
            return
        ds = self.get_entrypoint(_id)  # this should trigger registration
        if _id not in super().list_info() and raises:
            msg = (
                f"Could not find info for dataset ID {_id=}. "
                f"While {_id=} is a valid entrypoint, loading it did not yield a "
                f"registered info function. Found entrypoint: {ds}"
            )
            raise KeyError(msg)

    @TX.override
    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)
        return super().get_info(query)

    @TX.override
    def get_info_at(self, query: str, key: str) -> T.Any:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)
        return super().get_info_at(query, key)

    @TX.override
    def list_info(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return super().list_info() | self.list_entrypoints()
