from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX
import os
import typing as T
from multiprocessing import RLock

import pytest
import typing_extensions as TX
import wandb

from unipercept import file_io

from unipercept.utils.iopath_webdav import (
    WebDAVPathHandler,
)

import os
import typing as T
from pathlib import Path

import typing_extensions as TX

from unipercept import file_io


def test_file_io_environ():
    path = file_io.get_local_path("//datasets/")
    assert path == str(Path(os.environ.get("UNICORE_DATASETS", "datasets")).resolve())

    path = file_io.get_local_path("//cache/")
    assert path == str(Path(os.environ.get("UNICORE_CACHE", "cache")).resolve())

    path = file_io.get_local_path("//output/")
    assert path == str(Path(os.environ.get("UNICORE_OUTPUT", "output")).resolve())


WANDB_LOCK = RLock()


@pytest.fixture()
def wandb_run(tmp_path):
    os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_DIR"] = str(tmp_path)
    os.environ["WANDB_API_KEY"] = "test"

    with WANDB_LOCK:
        if wandb.run is None:
            wandb.init(project="test", entity="test")
    yield wandb.run
    # wandb.finish()


@pytest.mark.parametrize(
    "path",
    [
        "wandb-artifact://test/artifact/name:version",
        "wandb-artifact://test/artifact/name:version/",
        "wandb-artifact://test/artifact/name:version/path/to/file",
        "wandb-artifact:///test/artifact/name:version/path/to/file",
    ],
)
def test_wandb_artifact(wandb_run, path):
    with pytest.raises(FileNotFoundError):
        file_io.get_local_path(path)


# Assuming these are the settings for your WebDAV server
WEBDAV_OPTIONS = {
    "webdav_hostname": "https://webdav.example.com",
    "webdav_login": "username",
    "webdav_password": "password",
}


@pytest.fixture(scope="class")
def webdav_handler():
    """
    PyTest fixture for creating a WebDAVPathHandler instance.
    """
    return WebDAVPathHandler(WEBDAV_OPTIONS)


@pytest.mark.usefixtures("webdav_handler")
class TestWebDAVPathHandler:
    # Test Initialization
    def test_init(self, webdav_handler):
        assert webdav_handler is not None

    # Test File Existence
    def test_exists(self, webdav_handler):
        assert webdav_handler._exists("/path/to/existing/file.txt") is True
        assert webdav_handler._exists("/path/to/nonexistent/file.txt") is False

    # Test Directory Check
    def test_isdir(self, webdav_handler):
        assert webdav_handler._isdir("/path/to/directory/") is True
        assert webdav_handler._isdir("/path/to/file.txt") is False

    # Test File Check
    def test_isfile(self, webdav_handler):
        assert webdav_handler._isfile("/path/to/file.txt") is True
        assert webdav_handler._isfile("/path/to/directory/") is False

    # Test List Directory
    def test_listdir(self, webdav_handler):
        contents = webdav_handler._listdir("/path/to/directory/")
        assert isinstance(contents, list)
        assert "file.txt" in contents

    # Test File Opening (Read)
    def test_open_read(self, webdav_handler):
        with webdav_handler._open("/path/to/existing/file.txt", "r") as f:
            content = f.read()
            assert content is not None

    # Test File Opening (Write)
    def test_open_write(self, webdav_handler):
        test_content = "Hello, WebDAV!"
        with webdav_handler._open("/path/to/write/file.txt", "w") as f:
            f.write(test_content)

        # Verify content
        with webdav_handler._open("/path/to/write/file.txt", "r") as f:
            content = f.read()
            assert content == test_content

    # Add more tests as needed, such as for delete, move, etc.
