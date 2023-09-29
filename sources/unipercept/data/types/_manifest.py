"""Types used to create a dataset manifest."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import (
    Generic,
    Mapping,
    NotRequired,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
    final,
)

# ----------------- #
# Resources on disk #
# ----------------- #


class FileResource(TypedDict):
    """Describes a datapoint on the disk."""

    path: str


_MetaType = TypeVar("_MetaType", contravariant=True)


class FileResourceWithMeta(FileResource, Generic[_MetaType]):
    """Describes a datapoint on the disk, with metadata."""

    meta: _MetaType


_FormatType = TypeVar("_FormatType", contravariant=True)


class FormatMeta(TypedDict, Generic[_FormatType]):
    format: _FormatType


# --------------------------- #
# Capture sources and formats #
# --------------------------- #


class LabelsFormat(StrEnum):
    """
    Enumerates the different formats of labels that are supported. Uses the name of
    the dataset that introduced the format.
    """

    CITYSCAPES = auto()
    CITYSCAPES_VPS = auto()
    KITTI = auto()
    VISTAS = auto()
    WILD_DASH = auto()
    TORCH = auto()


PanopticResource: TypeAlias = FileResourceWithMeta[FormatMeta[LabelsFormat]]


class DepthFormat(StrEnum):
    DEPTH_INT16 = auto()


DepthResource: TypeAlias = FileResourceWithMeta[FormatMeta[DepthFormat]]


@final
class CaptureSources(TypedDict):
    """Paths to where files for this dataset may be found."""

    image: FileResource
    panoptic: NotRequired[FileResourceWithMeta[FormatMeta[LabelsFormat]]]
    depth: NotRequired[FileResourceWithMeta[FormatMeta[DepthFormat]]]


@final
class CaptureRecord(TypedDict):
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
    observer: NotRequired[str]


# -------------------------- #
# Motion sources and formats #
# -------------------------- #


@final
class MotionSources(TypedDict):
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
    observer: NotRequired[str]


@final
class MotionRecord(TypedDict):
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


class PinholeModelParameters(TypedDict):
    """Currently only supports Pinhole camera models."""

    focal_length: tuple[float, float]  # (fx, fy)
    principal_point: tuple[float, float]  # (cx, cy)
    rotation: tuple[float, float, float]  # (pitch (x), yaw (y), roll (z))
    translation: tuple[float, float, float]  # (tx, ty, tz)
    image_size: tuple[int, int]  # (height, width)


CameraModelParameters: TypeAlias = PinholeModelParameters

# ------------------- #
# Manifest of records #
# ------------------- #


@final
class ManifestSequence(TypedDict):
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

    camera: CameraModelParameters
    fps: float
    captures: Sequence[CaptureRecord]
    motions: NotRequired[Sequence[MotionRecord]]


@final
class Manifest(TypedDict):
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
    """

    timestamp: str
    version: str
    sequences: Mapping[str, ManifestSequence]


# ---------------- #
# Queue of records #
# ---------------- #


class QueueItem(TypedDict):
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
    observer: NotRequired[str]
    camera: CameraModelParameters
    captures: Sequence[CaptureSources]
    motions: NotRequired[Sequence[MotionSources]]
