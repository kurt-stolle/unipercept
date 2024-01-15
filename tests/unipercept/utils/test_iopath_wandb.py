import os
from multiprocessing import RLock

import pytest

import wandb
from unipercept import file_io

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
