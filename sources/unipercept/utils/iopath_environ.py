"""
Implements IOPath handlers for working with environment variables.
"""
from __future__ import annotations

import os
import os.path
import typing as T
import warnings
from pathlib import Path as _PathlibPath

import typing_extensions as TX
from iopath.common.file_io import PathHandler

__all__ = ["EnvironPathHandler"]


class EnvironPathHandler(PathHandler):
    """
    PathHandler that uses an environment variable to get the path.

    Parameters
    ----------
    prefix : str
        The prefix to use for this path handler.
    env : str
        The name of the environment variable to use.
    default : str, optional
        The default value to use if the environment variable is not defined, by default None.

    Raises
    ------
    ValueError
        If the environment variable is not defined and no default is provided.

    Examples
    --------
    >>> import os
    >>> os.environ["UNICORE_DATASETS"] = "/datasets"
    >>> handler = EnvPathHandler("//datasets/", "UNICORE_DATASETS")
    >>> handler.get_local_path("//datasets/foo/bar.txt")
    '/datasets/foo/bar.txt'
    """

    def __init__(self, prefix: str, env: str, default: str | None = None):
        value = os.getenv(env)
        if value is None or len(value) == 0 or value[0] == "-":
            if default is None:
                raise ValueError(f"Environment variable {env} not defined!")
            warnings.warn(f"Environment variable {env} not defined, using default {default!r}.", stacklevel=2)
            value = default

        value = os.path.expanduser(value)
        value = os.path.realpath(value)

        os.makedirs(value, exist_ok=True)

        self.PREFIX: T.Final = prefix
        self.LOCAL: T.Final = value

    @TX.override
    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_path(self, path: str, **kwargs) -> _PathlibPath:
        name = path[len(self.PREFIX) :]
        if len(name) == 0:
            return _PathlibPath(self.LOCAL).resolve()
        else:
            return _PathlibPath(self.LOCAL, *name.split("/")).resolve()

    @TX.override
    def _get_local_path(self, path: str, **kwargs):
        return str(self._get_path(path, **kwargs))

    @TX.override
    def _isfile(self, path: str, **kwargs: T.Any) -> bool:
        return self._get_path(path, **kwargs).is_file()

    @TX.override
    def _isdir(self, path: str, **kwargs: T.Any) -> bool:
        return self._get_path(path, **kwargs).is_dir()

    @TX.override
    def _ls(self, path: str, **kwargs: T.Any) -> list[str]:
        return sorted(p.name for p in self._get_path(path, **kwargs).iterdir())

    @TX.override
    def _open(self, path: str, mode="r", **kwargs):
        # name = path[len(self.PREFIX) :]
        # return _g_manager.open(self.LOCAL + name, mode, **kwargs)
        return open(self._get_local_path(path), mode, **kwargs)
