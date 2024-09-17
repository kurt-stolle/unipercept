"""
Implements user-facing API for functions that deal with data loading and management.
"""

from __future__ import annotations

import typing as T
import warnings

if T.TYPE_CHECKING:
    import unipercept.data.sets as unisets

__all__ = ["get_dataset", "get_info", "get_info_at", "list_datasets", "list_info"]


def __dir__() -> list[str]:
    return __all__


if T.TYPE_CHECKING:

    @T.overload
    def get_dataset(
        query: T.Literal["cityscapes"],  # noqa: U100
    ) -> type[unisets.cityscapes.CityscapesDataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["cityscapes-vps"],  # noqa: U100
    ) -> type[unisets.cityscapes.CityscapesVPSDataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["kitti-360"],  # noqa: U100
    ) -> type[unisets.kitti_360.KITTI360Dataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["kitti-step"],  # noqa: U100
    ) -> type[unisets.kitti_step.KITTISTEPDataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["kitti-sem"],  # noqa: U100
    ) -> type[unisets.kitti_sem.SemKITTIDataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["vistas"],  # noqa: U100
    ) -> type[unisets.vistas.VistasDataset]: ...

    @T.overload
    def get_dataset(
        query: T.Literal["wilddash"],  # noqa: U100
    ) -> type[unisets.wilddash.WildDashDataset]: ...

    @T.overload
    def get_dataset(
        query: str,  # noqa: U100
    ) -> type[unisets.PerceptionDataset]: ...

    @T.overload
    def get_dataset(
        query: None,  # noqa: U100
        **kwargs: T.Any,  # noqa: U100
    ) -> type[unisets.PerceptionDataset]: ...


def get_dataset(
    query: str | None = None, **kwargs: T.Any
) -> type[unisets.PerceptionDataset]:
    """
    Read a dataset from the catalog, returning the dataset **class** type.
    """
    from unipercept.data.sets import catalog

    if "name" in kwargs:
        assert query is None
        query = kwargs.pop("name")
        msg = "The 'name' argument is deprecated, use 'query' instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if len(kwargs) > 0:
        msg = f"Unexpected keyword arguments: {', '.join(kwargs)}"
        raise TypeError(msg)
    if query is None:
        msg = "No dataset query provided."
        raise ValueError(msg)

    return catalog.get(query)


def get_info(query: str) -> unisets.Metadata:
    """
    Read a dataset's info function from the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.get_info(query)


def get_info_at(query: str, key: str) -> unisets.Metadata:
    """
    Read a dataset's info function from the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.get_info_at(query, key=key)


def list_datasets() -> list[str]:
    """
    List the datasets registered in the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.list_datasets()


def list_info() -> list[str]:
    """
    List the info functions registered in the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.list_info()
