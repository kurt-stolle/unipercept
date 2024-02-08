"""
Implements IOPath handlers for working with environment variables.
"""
from __future__ import annotations

import os
import os.path
import tempfile
import typing as T
import warnings
from pathlib import Path as _PathlibPath
from urllib.parse import urlparse

import typing_extensions as TX
from iopath import file_lock, get_cache_dir
from iopath.common.file_io import PathHandler

from unipercept.log import get_logger

if T.TYPE_CHECKING:
    from wandb import Artifact as WandBArtifact

_logger = get_logger(__name__)

__all__ = [
    "WebDAVPathHandler",
    "WebDAVOptions",
    "EnvironPathHandler",
    "WandBArtifactHandler",
]


class WandBArtifactHandler(PathHandler):
    """
    Handles pulling artifacts from W&B using the API. Currently only supports reading.
    """

    def __init__(self, *, use_api: bool = True, use_run: bool = True):
        super().__init__()
        self.use_api = use_api
        self.use_run = use_run
        self.cache_map: dict[str, str] = {}

    def _parse_path(self, path: str) -> tuple[str, str | None]:
        """
        Format is one of the following:
         - wandb-artifact:///entity/project/name:version/file.h5
         - wandb-artifact:///entity/project/name:version
         - wandb-artifact://project/name:version/file.h5
        """
        url = urlparse(path)

        assert url.scheme == "wandb-artifact", f"Unsupported scheme {url.scheme!r}"

        # Spit by : to get name and combined version/file
        name, version_file = url.path.split(":")

        # Split by / to get version and filepath
        version, *maybe_file = version_file.split("/", 1)
        if len(maybe_file) > 0:
            file = maybe_file[0]
        else:
            file = None

        if len(url.netloc) > 0:
            name = f"{url.netloc}/{name}"
        elif name.startswith("/"):
            name = name[1:]

        name = name.strip("/")
        name = name.replace("//", "/")
        name = f"{name}:{version}"

        # Name is the netloc + name, where netloc could be empty
        return name, file

    def _get_artifact(self, name: str) -> WandBArtifact:
        import wandb

        if self.use_run and wandb.run is not None:
            _logger.debug("Using W&B artifact within a run")
            return wandb.run.use_artifact(name)
        elif self.use_api:
            _logger.debug("Using W&B artifact using the API (without a run)")
            return wandb.Api().artifact(name)
        else:
            raise RuntimeError("No W&B run or API available")

    @TX.override
    def _get_supported_prefixes(self) -> list[str]:
        return ["wandb-artifact://"]

    @TX.override
    def _get_local_path(
        self,
        path: str,
        mode: str = "r",
        force: bool = False,
        cache_dir: str | None = None,
        **kwargs,
    ):
        import wandb.errors

        self._check_kwargs(kwargs)

        assert mode in ("r",), f"Unsupported mode {mode!r}"

        if (
            force
            or path not in self.cache_map
            or not os.path.exists(self.cache_map[path])
        ):
            name, file = self._parse_path(path)

            try:
                artifact = self._get_artifact(name)
            except wandb.errors.CommError as e:
                raise FileNotFoundError(f"Could not find artifact {name!r}") from e

            path = os.path.join(get_cache_dir(cache_dir), name)
            with file_lock(path):
                if not os.path.exists(path) or force:
                    path = artifact.download(path)
                elif os.path.isfile(path):
                    raise FileExistsError(f"A file exists at {path!r}")
            path = os.path.join(path, file) if file is not None else path

            self.cache_map[path] = path
        return self.cache_map[path]

    @TX.override
    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any
    ) -> T.IO[str] | T.IO[bytes]:
        """
        Open a remote HTTP path. The resource is first downloaded and cached
        locally.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)

        if mode not in ("r",):
            raise ValueError(f"Unsupported mode {mode!r}")

        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path, force=False)
        return open(local_path, mode)


class WebDAVOptions(T.TypedDict):
    """
    Options to configure WebDAVPathHandler.
    """

    webdav_hostname: str
    webdav_login: str
    webdav_password: str


class WebDAVPathHandler(PathHandler):
    """
    PathHandler that uses WebDAV to access files.

    Parameters
    ----------
    webdav_options : dict

    """

    def __init__(self, webdav_options: WebDAVOptions):
        from webdav3.client import Client

        super().__init__()
        self.client = Client(webdav_options)

    @TX.override
    def _get_supported_prefixes(self) -> list[str]:
        return ["webdav://"]

    @TX.override
    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any
    ) -> T.Union[T.IO[str], T.IO[bytes]]:
        if mode not in ["r", "rb", "w", "wb"]:
            raise ValueError(f"Mode {mode} not supported for WebDAVPathHandler")

        local_path = self._download_to_local(path, mode)
        return open(local_path, mode, buffering, **kwargs)

    @TX.override
    def _exists(self, path: str, **kwargs: T.Any) -> bool:
        return self.client.check(path)

    @TX.override
    def _isfile(self, path: str, **kwargs: T.Any) -> bool:
        info = self.client.info(path)
        return info is not None and not info["isdir"]

    @TX.override
    def _isdir(self, path: str, **kwargs: T.Any) -> bool:
        info = self.client.info(path)
        return info is not None and info["isdir"]

    def _listdir(self, path: str, **kwargs: T.Any) -> list[str]:
        return [item["name"] for item in self.client.list(path)]

    def _download_to_local(self, path: str, mode: str) -> str:
        if "r" in mode:
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, os.path.basename(path))
            self.client.download_file(remote_path=path, local_path=local_path)
            return local_path
        elif "w" in mode:
            return os.path.join(tempfile.mkdtemp(), os.path.basename(path))

    def _upload_from_local(self, local_path: str, remote_path: str) -> None:
        self.client.upload_file(local_path=local_path, remote_path=remote_path)

    def _remove(self, path: str, **kwargs: T.Any) -> None:
        self.client.clean(path)

    def _mkdir(self, path: str, **kwargs: T.Any) -> None:
        self.client.mkdir(path)


class EnvironPathHandler(PathHandler):
    """
    PathHandler that uses an environment variable to get the path.

    Parameters
    ----------
    prefix : str
        The prefix to use for this path handler.
    env : str
        The name of the environment variable to use.
    default : str | None, optional
        The default value to use if the environment variable is not defined, by default None.
        If None is passed, then a temporary directory is created.

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

    def __init__(self, prefix: str, *env: str, default: str | None = None):
        self._tmp = None
        for env_key in env:
            value = os.getenv(env_key)
            if value is None or len(value) == 0 or value[0] == "-":
                continue
            else:
                break
        else:
            if default is None:
                self._tmp = tempfile.TemporaryDirectory()
                value = self._tmp.name
            else:
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

    def __del__(self):
        if self._tmp is not None:
            print(f"Removing temporary directory {self.PREFIX!r} at {self.LOCAL!r}")
            self._tmp.cleanup()
