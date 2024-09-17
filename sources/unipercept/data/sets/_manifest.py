r"""
Types used to create a dataset manifest.
"""

from __future__ import annotations

import typing as T

from unipercept.data.tensors._depth import DepthFormat
from unipercept.data.tensors._panoptic import LabelsFormat

__all__ = [
    "Manifest",
    "ManifestSequence",
    "Manifest",
    "ManifestSequence",
    "DepthMeta",
    "PanopticMeta",
    "QueueItem",
    "QueueGenerator",
    "CaptureRecord",
    "CaptureSources",
    "CaptureSourceKey",
    "MotionRecord",
    "MotionSources",
    "PinholeModelParameters",
    "CameraModelParameters",
    "FileResource",
    "FileResourceWithMeta",
]


# ----------------- #
# Resources on disk #
# ----------------- #


class FileResource(T.TypedDict):
    """Describes a datapoint on the disk."""

    path: str | list[str]


class FileResourceWithMeta[_M: T.Mapping](FileResource):
    """Describes a datapoint on the disk, with metadata."""

    meta: _M


class FormatMeta[_F](T.TypedDict):
    format: _F | str


class DepthMeta(FormatMeta[DepthFormat]):
    focal_length: T.NotRequired[float]


type PanopticMeta = FormatMeta[LabelsFormat]


# --------------------------- #
# Capture sources and formats #
# --------------------------- #

PanopticResource: T.TypeAlias = FileResourceWithMeta[FormatMeta[str]]
DepthResource: T.TypeAlias = FileResourceWithMeta[FormatMeta[str]]


@T.final
class CaptureSources(T.TypedDict):
    """Paths to where files for this dataset may be found."""

    image: FileResource
    panoptic: T.NotRequired[FileResourceWithMeta[PanopticMeta]]
    instance: T.NotRequired[FileResourceWithMeta[PanopticMeta]]
    semantic: T.NotRequired[FileResourceWithMeta[PanopticMeta]]
    depth: T.NotRequired[FileResourceWithMeta[DepthMeta]]


CaptureSourceKey: T.TypeAlias = T.Literal["image", "depth", "panoptic", "semantic"]


class CaptureRecord[_M: T.Mapping](T.TypedDict):
    """
    A record of captured data that is part of a temporal sequence.

    Parameters
    ----------
    primary_key: str
        Unique identifier of the record.
    sources : _CaptureSourceType
        Sources for data of the record.
    observer
        Name of the observer, if applicable. For example, the name of the camera that captured the motion.
        This can be useful when multiple cameras are used, like a stereo-setup.
    """

    primary_key: str
    sources: CaptureSources
    time: T.NotRequired[float]
    observer: T.NotRequired[str]
    meta: T.NotRequired[_M]


# -------------------------- #
# Motion sources and formats #
# -------------------------- #


@T.final
class MotionSources(T.TypedDict):
    """
    Paths to where files for this dataset may be found.

    Parameters
    ----------
    optical_flow
        Optical flow between the frames of the motion.
    transforms
        Transformations between the frames of the motion (in world coordinates).
    observer
        Name of the observer, if applicable. For example, the name of the camera that captured the motion.
        This can be useful when multiple cameras are used, like a stereo-setup.
    """

    optical_flow: FileResource
    transforms: FileResource


class MotionRecord[_M: T.Mapping](T.TypedDict):
    """
    A record of motion data in a temporal sequence.

    Parameters
    ----------
    frames:
        Frame numbers of the motion start and finish.
    sources
        Sources for data of the record.
    """

    frames: tuple[int, int]
    sources: MotionSources
    observer: T.NotRequired[str]
    meta: T.NotRequired[_M]


# ----------------- #
# Camera parameters #
# ----------------- #


class PinholeModelParameters(T.TypedDict):
    """
    Pinhole camera model
    """

    focal_length: tuple[float, float]  # (fx, fy)
    principal_point: tuple[float, float]  # (cx, cy)
    rotation: tuple[float, float, float]  # (pitch (x), yaw (y), roll (z))
    translation: tuple[float, float, float]  # (tx, ty, tz)
    image_size: tuple[int, int]  # (height, width)
    convention: T.NotRequired[str]  # 'opencv' (default) or 'opengl'
    observer: T.NotRequired[str]


type CameraModelParameters = PinholeModelParameters
type CameraResource = FileResourceWithMeta[FormatMeta[str]]

# ------------------- #
# Manifest of records #
# ------------------- #


class ManifestSequence[_M: T.Mapping](T.TypedDict):
    """
    Represents data about a single sequence in the manifest.

    Parameters
    ----------
    camera
        Camera parameters for the sequence (intrinsic and extrinsic matrix).
        When a dict is used, it corresponds to each 'observer' name in the
        captures and motions.
    fps
        Amount of frames per second. Drift is not supported, so this is assumed to be constant.
    captures
        Captures that are part of the item. Organized as a mapping from the sequence key to the captures in that
        sequence.
    motions
        Motions that are part of the item. Organized as a mapping from the sequence key to an unordered list of motions.
        Remember that each motion references a range of frames, and thus the order of the motions is not important.
    meta
        Metadata about the sequence.

    """

    camera: (
        CameraResource
        | list[CameraResource]
        | CameraModelParameters
        | list[CameraModelParameters]
        | None
    )
    fps: float | None
    captures: list[CaptureRecord]
    motions: T.NotRequired[list[MotionRecord]]
    meta: T.NotRequired[_M]


class Manifest[_M: T.Mapping](T.TypedDict):
    """
    A manifest of the dataset, before gathering the individual capture and motion records into
    the queued format that ought to be converted to the input data of the model.

    A manifest object essentially describes *all available data* for a given dataset.

    Parameters
    ----------
    timestamp
        Timestamp of when this manifest was created (for caching and invalidation purposes).
    version
        Version of Unipercept that was used to create this manifest (for caching and invalidation purposes).
    sequences
        Sequences that are part of the manifest. Organized as a mapping from the sequence key to the sequence.
    """

    timestamp: str
    version: str
    sequences: dict[str, ManifestSequence[_M]]


# ---------------- #
# Queue of records #
# ---------------- #


class QueueItem(T.TypedDict):
    """
    An item in the queue of the dataset, after gathering the individual capture and motion records into
    the format that ought to be converted to the input data of the model.

    A queue item is thus a single item in the iterable that is retuned by the queueing function.

    Parameters
    ----------
    sequence
        Unique identifier of the sequence.
    frame
        Frame number in the sequence of the first capture (and equivalently where the first motion starts).
    fps
        Amount of frames per second. Drift is not supported, so this is assumed to be constant.
    observer
        Name of the observer, if applicable. For example, the name of the camera that captured the motion.
        This can be useful when multiple cameras are used, like a stereo-setup.
    captures
        Captures that are part of the item, ordered by time.
    motions
        Motions that are part of the item, with the first element starting at the first capture and ending at the last
        capture. All motions must have a constant delta (i.e. all motions span the same amount of time)
    """

    sequence: str
    frame: int
    fps: float
    observer: T.NotRequired[str]
    camera: CameraModelParameters
    captures: list[CaptureSources]
    motions: T.NotRequired[list[MotionSources]]


type QueueGenerator = T.Iterable[tuple[str, QueueItem]]
