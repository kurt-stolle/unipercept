"""
Implements IOPath handlers for W&B.
"""

from __future__ import annotations

import os
import typing as T
from urllib.parse import urlparse

import typing_extensions as TX
from iopath import file_lock, get_cache_dir
from iopath.common.file_io import PathHandler

if T.TYPE_CHECKING:
    from wandb import Artifact as WandBArtifact

__all__ = ["WandBArtifactHandler"]


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

        name = f"{name}:{version}"

        # Name is the netloc + name, where netloc could be empty
        return name, file

    def _get_artifact(self, name: str) -> WandBArtifact:
        import wandb

        if self.use_run and wandb.run is not None:
            return wandb.run.use_artifact(name)
        elif self.use_api:
            return wandb.Api().artifact(name)
        else:
            raise RuntimeError("No W&B run or API available")

    @TX.override
    def _get_supported_prefixes(self) -> list[str]:
        return ["wandb-artifact://"]

    @TX.override
    def _get_local_path(self, path: str, mode: str = "r", force: bool = False, cache_dir: str | None = None, **kwargs):
        import wandb.errors

        self._check_kwargs(kwargs)

        assert mode in ("r",), f"Unsupported mode {mode!r}"

        if force or path not in self.cache_map or not os.path.exists(self.cache_map[path]):
            name, file = self._parse_path(path)

            try:
                artifact = self._get_artifact(name)
            except wandb.errors.CommError as e:
                raise FileNotFoundError(f"Could not find artifact {name!r}") from e

            path = os.path.join(get_cache_dir(cache_dir), name)
            with file_lock(path):
                if not os.path.exists(path) or force:
                    path = artifact.checkout(path)
                elif os.path.isfile(path):
                    raise FileExistsError(f"A file exists at {path!r}")
            path = os.path.join(path, file) if file is not None else path

            self.cache_map[path] = path
        return self.cache_map[path]

    @TX.override
    def _open(self, path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any) -> T.IO[str] | T.IO[bytes]:
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

        assert buffering == -1, f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path, force=False)
        return open(local_path, mode)
