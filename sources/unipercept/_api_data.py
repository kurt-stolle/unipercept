"""
Implements user-facing API for functions that deal with data loading and management.
"""

from __future__ import annotations

import typing as T

if T.TYPE_CHECKING:
    import unipercept.data.sets as unisets

__all__ = ["get_dataset", "get_info", "list_datasets", "list_info"]


@T.overload
def get_dataset(name: T.Literal["cityscapes"]) -> type[unisets.cityscapes.CityscapesDataset]:
    ...


@T.overload
def get_dataset(name: T.Literal["cityscapes-vps"]) -> type[unisets.cityscapes.CityscapesVPSDataset]:
    ...


@T.overload
def get_dataset(name: T.Literal["kitti-360"]) -> type[unisets.kitti_360.KITTI360Dataset]:
    ...


@T.overload
def get_dataset(name: T.Literal["kitti-sem"]) -> type[unisets.kitti_sem.SemKITTIDataset]:
    ...


@T.overload
def get_dataset(name: T.Literal["vistas"]) -> type[unisets.vistas.VistasDataset]:
    ...


@T.overload
def get_dataset(name: T.Literal["wilddash"]) -> type[unisets.wilddash.WildDashDataset]:
    ...


def get_dataset(name: str) -> type[unisets.PerceptionDataset]:
    """
    Read a dataset from the catalog, returning the dataset **class** type.
    """
    from unipercept.data.sets import catalog

    return catalog.get_dataset(name)


def get_info(name: str) -> unisets.Metadata:
    """
    Read a dataset's info function from the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.get_info(name)


def list_datasets() -> T.List[str]:
    """
    List the datasets registered in the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.list_datasets()


def list_info() -> T.List[str]:
    """
    List the info functions registered in the catalog.
    """
    from unipercept.data.sets import catalog

    return catalog.list_info()
