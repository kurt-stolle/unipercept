"""
Extends `pathlib.Path` to work with `iopath.common.file_io.PathManager`.
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from iopath.common.file_io import PathManager
from typing_extensions import Self, override

__all__ = ["IoPath"]


if TYPE_CHECKING:
    _Path = Path
else:
    _Path = type(Path(__file__))


class IoPath(_Path):
    """
    Extends `pathlib.Path` to work with `iopath.common.file_io.PathManager`.

    When manipulated with `pathlib.Path`, the object will be converted to a local path using
    `PathManager.get_local_path`.

    This class is not meant to be instantiated directly. Instead, use a subclass that specifies a `manager`
    class variable.
    """

    _manager: ClassVar[PathManager]

    def __new__(cls, path: str | os.PathLike, *args, force=False):
        if cls is IoPath:
            raise TypeError("Cannot instantiate IoPath directly, use a subclass instead")
        if isinstance(path, str):
            path = cls._manager.get_local_path(path, force=force)
        return cast(Self, _Path(path, *args))

    @override
    def __init_subclass__(cls, /, manager: PathManager, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._manager = manager

    def __getattr__(self, name: str) -> Any:
        """
        Forward all other attribute accesses to the underlying `pathlib.Path` object.
        """
        return getattr(Path, name)
