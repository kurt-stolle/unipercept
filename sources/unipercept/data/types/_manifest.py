"""Types used to create a dataset manifest."""

from __future__ import annotations

import typing as T

if T.TYPE_CHECKING:
    from ..tensors import DepthFormat, LabelsFormat
else:
    DepthFormat = T.Any
    LabelsFormat = T.Any

__all__ = [
    "Manifest",
    "ManifestSequence",
    "Manifest",
    "ManifestSequence",
    "QueueItem",
    "CaptureRecord",
    "CaptureSources",
    "MotionRecord",
    "MotionSources",
    "PinholeModelParameters",
    "FileResource",
    "FileResourceWithMeta",
]


# ----------------- #
# Resources on disk #
# ----------------- #


class FileResource(T.TypedDict):
    """Describes a datapoint on the disk."""

    path: str


_MetaType = T.TypeVar("_MetaType", bound=T.TypedDict, contravariant=True)


class FileResourceWithMeta(FileResource, T.Generic[_MetaType]):
    """Describes a datapoint on the disk, with metadata."""

    meta: _MetaType


_FormatType = T.TypeVar("_FormatType", contravariant=True)


class FormatMeta(T.TypedDict, T.Generic[_FormatType]):
    format: _FormatType | str


class DepthMeta(FormatMeta[DepthFormat]):
    focal_length: T.NotRequired[float]


# --------------------------- #
# Capture sources and formats #
# --------------------------- #

PanopticResource: T.TypeAlias = FileResourceWithMeta[FormatMeta[LabelsFormat]]
DepthResource: T.TypeAlias = FileResourceWithMeta[FormatMeta[DepthFormat]]


@T.final
class CaptureSources(T.TypedDict):
    """Paths to where files for this dataset may be found."""

    image: FileResource
    panoptic: T.NotRequired[FileResourceWithMeta[FormatMeta[LabelsFormat]]]
    instance: T.NotRequired[FileResourceWithMeta[FormatMeta[LabelsFormat]]]
    semantic: T.NotRequired[FileResourceWithMeta[FormatMeta[LabelsFormat]]]
    depth: T.NotRequired[FileResourceWithMeta[DepthMeta]]


@T.final
class CaptureRecord(T.TypedDict, T.Generic[_MetaType]):
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
    meta: T.NotRequired[_MetaType]


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
    observer: T.NotRequired[str]


@T.final
class MotionRecord(T.TypedDict):
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


CameraModelParameters: T.TypeAlias = PinholeModelParameters

# ------------------- #
# Manifest of records #
# ------------------- #


@T.final
class ManifestSequence(T.TypedDict):
    """
    Represents data about a single sequence in the manifest.

    Parameters
    ----------
    camera
        Camera parameters for the sequence (intrinsic and extrinsic matrix).
    fps
        Amount of frames per second. Drift is not supported, so this is assumed to be constant.
    captures
        Captures that are part of the item. Organized as a mapping from the sequence key to the captures in that
        sequence.
    motions
        Motions that are part of the item. Organized as a mapping from the sequence key to an unordered list of motions.
        Remember that each motion references a range of frames, and thus the order of the motions is not important.
    """

    camera: CameraModelParameters | None
    fps: float | None
    captures: list[CaptureRecord]
    motions: T.NotRequired[list[MotionRecord]]


@T.final
class Manifest(T.TypedDict):
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
    sequences: dict[str, ManifestSequence]


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
