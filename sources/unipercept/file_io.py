"""
Implements a path manager for UniCore using IoPath.
"""

from __future__ import annotations

import functools
import typing as T

import typing_extensions as TX
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler, PathManager, PathManagerFactory

from unipercept.utils.iopath_environ import EnvironPathHandler
from unipercept.utils.iopath_path import IoPath
from unipercept.utils.iopath_wandb import WandBArtifactHandler

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


#################
# Path handlers #
#################
# Register handlers with the manager object
for h in (
    OneDrivePathHandler(),
    HTTPURLHandler(),
    WandBArtifactHandler(),
    EnvironPathHandler("//datasets/", "UNICORE_DATASETS", "./datasets"),
    EnvironPathHandler("//cache/", "UNICORE_CACHE", "~/.torch/unicore/cache"),
    EnvironPathHandler("//output/", "UNICORE_OUTPUT", "./output"),
    EnvironPathHandler("//scratch/", "UNICORE_SCRATCH", "./scratch"),
):
    _manager.register_handler(h, allow_override=False)
_exports: frozenset[str] = frozenset(fn_name for fn_name in dir(_manager) if not fn_name.startswith("_"))

##############
# Decorators #
##############

_Params = T.ParamSpec("_Params")
_Return = T.TypeVar("_Return")
_PathCallable: T.TypeAlias = T.Callable[T.Concatenate[str, _Params], _Return]


def with_local_path(
    fn: _PathCallable | None = None,
    *,
    manager: PathManager = _manager,
    **get_local_path_kwargs: T.Any,
) -> _PathCallable | T.Callable[[_PathCallable], _PathCallable]:
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
    def Wrapper(path: str, *args: _Params.args, **kwargs: _Params.kwargs):
        path = manager.get_local_path(path, **get_local_path_kwargs)
        return fn(path, *args, **kwargs)

    return Wrapper


#################################
# Exported methods from manager #
#################################


def __getattr__(name: str):
    if name in _exports:
        return getattr(_manager, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    global _exports
    return __all__ + list(_exports)


if T.TYPE_CHECKING:

    def opent(path: str, mode: str = "r", buffering: int = 32, **kwargs: T.Any) -> T.Iterable[T.Any]:
        ...

    def open(path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any) -> IO[str] | T.IO[bytes]:
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

    def copy(src_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any) -> bool:
        ...

    def mv(src_path: str, dst_path: str, **kwargs: T.Any) -> bool:
        ...

    def get_local_path(path: str, force: bool = False, **kwargs: T.Any) -> str:
        ...

    def copy_from_local(local_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any) -> None:
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
