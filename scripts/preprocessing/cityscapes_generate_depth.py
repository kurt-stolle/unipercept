"""
Generates depth maps from the Cityscapes dataset disparity maps.

The disparity maps are converted to depth maps using the following formula:
    depth = (f * B) / disparity
where f is the focal length and B is the baseline.

Camera files are read from the 'camera' subdirectory of the dataset root directory.
The JSON files have format as in the following example:
```
{
    "extrinsic": {
        "baseline": 0.222126, 
        "pitch": 0.05, 
        "roll": 0.0, 
        "x": 1.7, 
        "y": -0.1, 
        "yaw": 0.007, 
        "z": 1.18
    }, 
    "intrinsic": {
        "fx": 2268.36, 
        "fy": 2225.5405988775956, 
        "u0": 1048.64, 
        "v0": 519.277
    }
}
```

Disparity files are read from the 'disparity' subdirectory of the dataset root directory.
The disparity maps are stored as 16-bit PNG files, where each pixel value is the disparity
multiplied by 256.
"""

import argparse
import multiprocessing
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import PIL
from tqdm import tqdm


class DriveID(NamedTuple):
    city: str
    drive: str


class FileID(NamedTuple):
    city: str
    drive: str
    frame: str


def file_id_to_string(file_id: FileID) -> str:
    return f"{file_id.city}_{file_id.drive}_{file_id.frame}"


def file_id_to_drive_id(file_id: FileID) -> DriveID:
    return DriveID(file_id.city, file_id.drive)


class CameraParameters(NamedTuple):
    baseline: float
    fx: float


FILE_NAME_PATTERN = re.compile(
    r"(?P<city>[A-Za-z]+)_" r"(?P<drive>\d\d\d\d\d\d)_" r"(?P<frame>\d\d\d\d\d\d)_" r"(?P<ext>.+)\..+$"  # noqa: 501
)


def match_id(path: str) -> tuple[FileID]:
    """
    Transforms a path into an ID and a dictionary of paths indexed by key.
    """

    match = FILE_NAME_PATTERN.search(path)
    assert match is not None
    return FileID(match.group("city"), match.group("drive"), match.group("frame"))


def get_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        help="Output directory (optional)",
    )
    p.add_argument(
        "root",
        type=Path,
        help=(
            "Cityscapes dataset root directory. Requires the disparity maps to be in the 'disparity' "
            "subdirectory and camera calibration files to be in the 'camera' subdirectory."
        ),
    )

    return p


def read_camera(path: Path) -> CameraParameters:
    import json

    with path.open("r") as f:
        data = json.load(f)
        return CameraParameters(
            baseline=data["extrinsic"]["baseline"],
            fx=data["intrinsic"]["fx"],
        )


def read_disparity(path: Path) -> np.ndarray:
    img = PIL.Image.open(path)
    assert img.mode == "I"
    return np.array(img, dtype=np.uint16) / 256.0


def write_depth(path: Path, depth: np.ndarray) -> None:
    img = PIL.Image.fromarray(depth.astype(np.uint16))
    img.save(path)


def process_image(path_disp: Path, path_cam: Path, path_out: Path) -> None:
    cam = read_camera(path_cam)
    disp = read_disparity(path_disp)
    depth = (cam.fx * cam.baseline) / disp
    write_depth(path_out, depth)


def main(dir_root: Path, dir_out: Path) -> None:
    dir_disp = dir_root / "disparity"
    if not dir_disp.is_dir():
        raise RuntimeError(f"Directory '{dir_disp}' does not exist")
    dir_cam = dir_root / "camera"
    if not dir_cam.is_dir():
        raise RuntimeError(f"Directory '{dir_cam}' does not exist")
    dir_out.mkdir(parents=True, exist_ok=False)

    # Create a mapping from file ID to disparity map path
    files_disp: dict[FileID, Path] = {}
    for disp_file in tqdm(dir_disp.glob("**/*.png"), desc="Reading disparity map IDs"):
        file_id = match_id(disp_file.name)
        files_disp[file_id] = disp_file

    # Create a mapping from (city, drive) to camera parameters path.
    # This is needed because the camera parameters are the same for all frames.
    files_cam: dict[DriveID, Path] = {}
    for cam_file in tqdm(dir_cam.glob("**/*.json"), desc="Reading camera parameters"):
        drive_id = file_id_to_drive_id(match_id(cam_file.name))
        files_cam[drive_id] = cam_file

    # Process each image in parallel
    with multiprocessing.Pool() as pool:
        tasks = []
        for file_id, path_disp in tqdm(files_disp.items(), desc="Matching disparity maps to calibration files"):
            drive_id = file_id_to_drive_id(file_id)
            path_cam = files_cam[drive_id]
            path_out = dir_out / f"{file_id_to_string(file_id)}.png"
            tasks.append((path_disp, path_cam, path_out))
        for _ in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing images"):
            pass

    print("Done")


if __name__ == "__main__":
    args = get_argparser().parse_args()

    if args.output is None:
        args.output = args.root / "depth"

    main(args.root, args.output)
