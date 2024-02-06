from __future__ import annotations
from dis import disco
from email.mime import image
from pathlib import Path
import re

import typing as T

import pytest
from tensordict import TensorDict
import torch
import typing_extensions as TX

import unipercept as up

import pytest
import torch
from unipercept.data.sets import Metadata


#################################
# Directories with testing data #
#################################


def test_data_root(tmp_path, request):
    """
    Root directory for the test data. Amounts to a temporary directory with the contents
    of the directory having the same name as the test module (i.e. filename) copied
    to it. Safe for reading and writing.
    """

    from distutils import dir_util

    root = Path(request.module.__file__)
    root = root.parent / root.stem
    tmp_path = tmp_path / "data"
    tmp_path.mkdirs()

    if root.exists() and root.is_dir():
        dir_util.copy_tree(str(root), str(tmp_path))

    return tmp_path


##############################################
# Mock inputs, ground truths and predictions #
##############################################


MOCK_DATA_ROOT = Path(__file__).parent.parent / "assets" / "testing"
MOCK_DATA_PATTERN = re.compile(r"(\d{4})/(\d{6}).png$")
MOCK_DATA_DEPTH_FORMAT = up.data.tensors.DepthFormat.DEPTH_INT16


def discover_mockfiles() -> T.Iterator[T.Tuple[int, int]]:
    """
    We use the provided testing assets directory for the mock data.
    Do not write to this directory.

    The structure is as follows:

    ```
    //root/
    ├── inputs/                 <-- input images
    │   ├── 0000/               <-- sequence ID
    │   │   ├── 000000.png      <-- frame ID
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── 0001/
    │       ├── frame1.png
    │       ├── frame2.png
    │       └── ...
    ├── truths/                 <-- ground truth data
    |   ├── segmentations/      <-- segmentations (panoptic)
    │   │   ├── 0000/
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── 0001/
    │   │       ├── frame1.png
    │   │       ├── frame2.png
    │   │       └── ...
    │   └── depths/             <-- depths
    │       ├── 0000/
    │       │   ├── 000000.png
    │       │   ├── 000001.png
    │       │   └── ...
    │       └── 0001/
    │           ├── 000000.png
    │           ├── 000001.png
    │           └── ...
    └── predictions/            <-- predicted data
        ├── segmentations/      <-- segmentations (panoptic)
        │   ├── 0000/
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── 0001/
        │       ├── 000000.png
        │       ├── 000001.png
        │       └── ...
        └── depths/             <-- depths
            ├── 0000/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── 0001/
                ├── 000000.png
                ├── 000001.png
                └── ...

    """
    for input_file in MOCK_DATA_ROOT.glob("inputs/*/*.png"):
        match = MOCK_DATA_PATTERN.search(input_file.as_posix())
        assert (
            match is not None
        ), f"Invalid input file: {input_file}; does not match pattern {MOCK_DATA_PATTERN.pattern}"

        seq_id, frame_id = match.group(1), match.group(2)
        yield int(seq_id), int(frame_id)


@pytest.fixture(scope="session")
def mock_info():
    """
    The true panoptic segmentation.
    """

    from unipercept import get_info

    return get_info("kitti-step")


@pytest.fixture(scope="session")
def mock_data(
    mock_info: Metadata,
) -> list[tuple[up.model.InputData, up.model.ModelOutput]]:
    """
    Mocks the data going in and out of a model.
    """

    inputs = [
        (
            up.model.InputData(
                ids=torch.tensor((seq_id, frame_id), dtype=torch.long),
                captures=up.model.CaptureData(
                    times=torch.tensor([frame_id / mock_info.fps], dtype=torch.float32),
                    images=up.data.tensors.Image.read(
                        MOCK_DATA_ROOT / f"inputs/{seq_id:04d}/{frame_id:06d}.png"
                    ),
                    segmentations=up.data.tensors.PanopticMap.read(
                        MOCK_DATA_ROOT
                        / f"truths/segmentations/{seq_id:04d}/{frame_id:06d}.png",
                        mock_info,
                    ),
                    depths=up.data.tensors.DepthMap.read(
                        MOCK_DATA_ROOT
                        / f"truths/depths/{seq_id:04d}/{frame_id:06d}.png",
                        format=MOCK_DATA_DEPTH_FORMAT,
                    ),
                ),
            ),
            TensorDict(
                {
                    "segmentations": up.data.tensors.PanopticMap.read(
                        MOCK_DATA_ROOT
                        / f"predictions/segmentations/{seq_id:04d}/{frame_id:06d}.png",
                        mock_info,
                    ),
                    "depths": up.data.tensors.DepthMap.read(
                        MOCK_DATA_ROOT
                        / f"predictions/depths/{seq_id:04d}/{frame_id:06d}.tiff",
                        format=up.data.tensors.DepthFormat.TIFF,
                    ),
                }
            ),
        )
        for seq_id, frame_id in discover_mockfiles()
    ]

    return inputs
