"""
Implements T.IOPath handlers for WebDAV.
"""

from __future__ import annotations

import os
import tempfile
import typing as T

import typing_extensions as TX
from iopath.common.file_io import PathHandler

__all__ = ["WebDAVPathHandler", "WebDAVOptions"]


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
