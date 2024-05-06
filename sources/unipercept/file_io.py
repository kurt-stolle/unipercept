"""
Implements a path manager using ``iopath``.
"""

from __future__ import annotations

import functools
import os
import typing as T

from iopath.common.file_io import (
    HTTPURLHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManager,
    PathManagerFactory,
)

from unipercept.utils.iopath_handlers import EnvironPathHandler, WandBArtifactHandler
from unipercept.utils.iopath_path import IoPath
from unipercept.utils.typings import Pathable

__all__ = ["Path"]


################
# Path manager #
################

_manager: T.Final = PathManagerFactory.get(defaults_setup=False)


#################
# Path subclass #
#################
class Path(IoPath, manager=_manager):
    """
    See ``IoPath``.
    """


#############
# Utilities #
#############


def join(base: str | Path, *other: str | Path) -> str:
    """
    Joins paths using the path manager.

    Parameters
    ----------
    *paths : str
        The paths to join.

    Returns
    -------
    str
        The joined path.

    """
    base = str(base)
    return os.path.join(base, *map(str, other))


#################
# Path handlers #
#################
# Register handlers with the manager object
for h in (
    OneDrivePathHandler(),
    HTTPURLHandler(),
    WandBArtifactHandler(),
    EnvironPathHandler(
        "//datasets/",
        "UP_DATASETS",
        "UNIPERCEPT_DATASETS",
        "UNICORE_DATASETS",
        "DETECTRON2_DATASETS",
        "D2_DATASETS",
        default="~/datasets",
    ),
    EnvironPathHandler(
        "//cache/",
        "UP_CACHE",
        "UNIPERCEPT_CACHE",
        "UNICORE_CACHE",
        default="~/.cache/unipercept",
    ),
    EnvironPathHandler(
        "//output/",
        "UP_OUTPUT",
        "UNIPERCEPT_OUTPUT",
        "UNICORE_OUTPUT",
        default="./output",
    ),
    EnvironPathHandler(
        "//configs/", "UP_CONFIGS", "UNIPERCEPT_CONFIGS", default="./configs"
    ),
    EnvironPathHandler(
        "//scratch/",
        "UP_SCRATCH",
        "UNIPERCEPT_SCRATCH",
        "UNICORE_SCRATCH",
        default=None,
    ),
):
    _manager.register_handler(h, allow_override=False)
_exports: frozenset[str] = frozenset(
    fn_name for fn_name in dir(_manager) if not fn_name.startswith("_")
)

##############
# Decorators #
##############

_Params = T.ParamSpec("_Params")
_Return = T.TypeVar("_Return")
_PathStrCallable: T.TypeAlias = T.Callable[T.Concatenate[str, _Params], _Return]
_PathAnyCallable: T.TypeAlias = T.Callable[T.Concatenate[Pathable, _Params], _Return]


@T.overload
def with_local_path(
    fn: None = None,
    *,
    manager: PathManager = _manager,
    **get_local_path_kwargs: T.Any,
) -> T.Callable[[_PathStrCallable], _PathAnyCallable]:
    ...


@T.overload
def with_local_path(
    fn: _PathStrCallable,
    *,
    manager: PathManager = _manager,
    **get_local_path_kwargs: T.Any,
) -> _PathAnyCallable:
    ...


def with_local_path(
    fn: _PathStrCallable | None = None,
    *,
    manager: PathManager = _manager,
    **get_local_path_kwargs: T.Any,
) -> _PathAnyCallable | T.Callable[[_PathStrCallable], _PathAnyCallable]:
    """
    Decorator that converts the first argument of a function to a local path.

    This is useful for functions that take a path as the first argument, but
    the path is not necessarily local. This decorator will convert the path
    to a local path using the path manager, and pass the result to the function.

    Parameters
    ----------
    fn : Callable
        The function to decorate.
    manager : PathManager, optional
        The path manager to use, by default the default path manager.
    **get_local_path_kwargs : Any
        Keyword arguments to pass to the path manager's ``get_local_path`` method.

    Returns
    -------
    Callable
        The decorated function.

    """

    if fn is None:
        return functools.partial(with_local_path, manager=manager, **get_local_path_kwargs)  # type: ignore

    @functools.wraps(fn)
    def Wrapper(path: Pathable, *args: _Params.args, **kwargs: _Params.kwargs):
        path = manager.get_local_path(str(path), **get_local_path_kwargs)
        return fn(path, *args, **kwargs)

    return Wrapper


#################################
# Exported methods from manager #
#################################


@with_local_path
def get_local_path(path: str, force: bool = False, **kwargs: T.Any) -> str:
    return _manager.get_local_path(path, force=force, **kwargs)


def __getattr__(name: str):
    if name in _exports:
        return getattr(_manager, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    global _exports
    return __all__ + list(_exports)


if T.TYPE_CHECKING:

    def opent(
        path: str, mode: str = "r", buffering: int = 32, **kwargs: T.Any
    ) -> T.Iterable[T.Any]:
        ...

    def open(
        path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any
    ) -> T.IO[str] | T.IO[bytes]:
        ...

    def opena(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        callback_after_file_close: T.Optional[T.Callable[[None], None]] = None,
        **kwargs: T.Any,
    ) -> T.IO[str] | T.IO[bytes]:
        ...

    def async_join(*paths: str, **kwargs: T.Any) -> bool:
        ...

    def async_close(**kwargs: T.Any) -> bool:
        ...

    def copy(
        src_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any
    ) -> bool:
        ...

    def mv(src_path: str, dst_path: str, **kwargs: T.Any) -> bool:
        ...

    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any
    ) -> None:
        ...

    def exists(path: str, **kwargs: T.Any) -> bool:
        ...

    def isfile(path: str, **kwargs: T.Any) -> bool:
        ...

    def isdir(path: str, **kwargs: T.Any) -> bool:
        ...

    def ls(path: str, **kwargs: T.Any) -> list[str]:
        ...

    def mkdirs(path: str, **kwargs: T.Any) -> None:
        ...

    def rm(path: str, **kwargs: T.Any) -> None:
        ...

    def symlink(src_path: str, dst_path: str, **kwargs: T.Any) -> bool:
        ...

    def set_cwd(path: T.Optional[str], **kwargs: T.Any) -> bool:
        ...

    def register_handler(handler: PathHandler, allow_override: bool = True) -> None:
        ...

    def set_strict_kwargs_checking(enable: bool) -> None:
        ...

    def set_logging(enable_logging=True) -> None:
        ...
