"""
Implements a simple data manager for registering datasets and their info functions.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Any, Final, cast, override

import regex as re

from unipercept.utils.registry import Registry

__all__ = ["Catalog", "CatalogFromPackageMetadata"]


_DEFAULT_ID_PATTERN: Final[re.Pattern] = re.compile(r"^[a-z\d\-]+$")
_DEFAULT_ID_SEPARATOR: Final[str] = "/"


class Catalog[_D_co, _I_co]:
    """
    Data manager for registering datasets and their info functions.
    """

    __slots__ = (
        "_variant_sep",
        "_id_regex",
        "_require_info",
        "_info_registry",
        "_data_registry",
    )

    def __init__(
        self,
        *,
        id_pattern: re.Pattern[str] = _DEFAULT_ID_PATTERN,
        variant_separator: str = _DEFAULT_ID_SEPARATOR,
        require_info: bool = True,
    ):
        """
        Parameters
        ----------
        id_pattern : re.Pattern
            The pattern to use for validating dataset IDs.
        variant_separator : str
            The separator to use for separating dataset IDs from variant IDs.
        """
        self._require_info: Final[bool] = require_info
        self._variant_sep: Final[str] = variant_separator
        self._id_regex: Final[re.Pattern[str]] = id_pattern

        self._info_registry = cast(
            Registry[Callable[..., _I_co], str], Registry(self.parse_key)
        )
        self._data_registry = cast(Registry[type[_D_co], str], Registry(self.parse_key))

    def parse_key(self, key: str | type[_D_co]) -> str:
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

        if not isinstance(key, str):
            if hasattr(key, "__name__"):
                name = key.__name__.lower()  # type: ignore
            else:
                msg = f"Cannot convert {key} to a canonical ID, as it has no name."
                raise ValueError(msg)
        else:
            name = key.lower()
        match = self._id_regex.search(name)
        if not match:
            raise ValueError(f"{key} ({name}) does not match {self._id_regex.pattern}")
        return match.group()

    def split_query(self, query: str) -> tuple[str, frozenset[str]]:
        """
        Split a query into a dataset ID and a variant ID.
        """
        if self._variant_sep not in query:
            return query, []
        key, *variant = query.split(self._variant_sep, maxsplit=1)
        return key, variant

    def __ior__(self, __other: Catalog, /) -> T.Self:
        """
        Merge the data and info registries of this manager with another.
        The other manager takes precedence in case of conflicts.
        """
        self._data_registry |= __other._data_registry
        self._info_registry |= __other._info_registry

        return self

    def __or__(self, __other: Catalog, /) -> T.Self:
        from copy import copy

        obj = copy(self)
        obj |= __other

        return obj

    def fork(self) -> Catalog:
        """
        Return a copy of this data manager.
        """
        return Catalog() | self

    # ------------- #
    # MAIN ELEMENTS #
    # ------------- #

    def register(
        self, id: str | None = None, *, info: Callable[..., _I_co] | None = None
    ) -> Callable[[type[_D_co]], type[_D_co]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : Optional[str]
            The ID to register the dataset with. If None, the dataset class name will be used (flattened and converted
            to snake_case).
        """

        if info is None and self._require_info:
            raise ValueError("No info function provided, but required.")

        def wrapped(ds: type[_D_co]) -> type[_D_co]:
            key = id or self.parse_key(ds)
            self._data_registry[key] = ds
            if callable(info):
                self._info_registry[key] = info
            elif self._require_info:
                raise TypeError(f"Invalid info function: {info}")

            return ds

        return wrapped

    def get(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        return self._data_registry[query]

    def list(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return frozenset(self._data_registry.keys())

    # ---- #
    # Info #
    # ---- #

    def register_info[**_P](
        self,
        key: str,
        /,
    ) -> Callable[[Callable[_P, _I_co]], Callable[_P, _I_co]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : Optional[str]
            The ID to register the dataset with. If None, the dataset class name will be canonicalized using
            ``canonicalize_id``.
        """

        def wrapped(info: Callable[_P, _I_co]) -> Callable[_P, _I_co]:
            self._info_registry[key] = info

            return info

        return wrapped

    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)

    def get_info_at(self, query: str, key: str) -> Any:
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


class CatalogFromPackageMetadata[_D_co, _I_co](Catalog[_D_co, _I_co]):
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

    def get_entrypoint(self, key: str) -> type[_D_co]:
        """
        Return the entrypoint for the given dataset ID.
        """
        try:
            return importlib.metadata.entry_points(group=self.group)[key].load()
        except KeyError:
            msg = f"Could not find entrypoint for dataset ID {key=}. Choose from: {self.list_entrypoints()}"
            raise KeyError(msg)

    @override
    def get(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        try:
            return self._data_registry[query]
        except KeyError:
            return self.get_entrypoint(query)

    @override
    def list(self) -> frozenset[str]:
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

    @override
    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)
        return super().get_info(query)

    @override
    def get_info_at(self, query: str, key: str) -> T.Any:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)
        return super().get_info_at(query, key)

    @override
    def list_info(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return super().list_info() | self.list_entrypoints()
