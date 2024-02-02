"""This module contains methods that can be used to generate a queue from a manifest."""

from __future__ import annotations

import typing as T

from typing_extensions import override

import unipercept.data.types as data_types
from unipercept.log import get_logger

__all__ = [
    "GroupAdjacentTime",
    "ExtractIndividualFrames",
    "QueueGeneratorType",
    "KnownCaptureSources",
]


_logger = get_logger(__name__)

QueueGeneratorType: T.TypeAlias = T.Generator[
    tuple[str, data_types.QueueItem], None, None
]
KnownCaptureSources: T.TypeAlias = T.Literal["image", "depth", "panoptic"]


class GroupAdjacentTime:
    """
    Queue that collects queue items of a specified length from the manifest, where items in the queue are temporally
    adjacent to each other. This is useful for tasks such as optical flow, where the input is a sequence of images.
    """

    _step_size: frozenset[int] | None
    _required_capture_sources: tuple[frozenset[str], ...]
    _num_frames: int
    _use_typecheck: bool

    def __init__(
        self,
        num_frames: int,
        *,
        use_typecheck=False,
        required_capture_sources: set[KnownCaptureSources | str]
        | T.Sequence[set[KnownCaptureSources | str]]
        | None = None,
        verbose: bool = False,
        step_size: set[int] | None = None,
    ):
        if num_frames <= 0:
            raise ValueError(f"Length must be positive definite, got {num_frames}!")

        if required_capture_sources is None:
            self._required_capture_sources = (frozenset({"image"}),) * num_frames
        elif all(isinstance(t, str) for t in required_capture_sources):
            self._required_capture_sources = (
                frozenset(map(str, required_capture_sources)),
            ) * num_frames
        elif isinstance(required_capture_sources, T.Iterable):
            self._required_capture_sources = tuple(
                map(frozenset, required_capture_sources)
            )
        elif len(required_capture_sources) != num_frames:
            raise ValueError(
                f"Expected {num_frames} truth requirements, got {len(required_capture_sources)}!"
            )
        else:
            raise TypeError(
                f"Expected a sequence of strings or a sequence of sequences of strings!"
            )

        self._num_frames = num_frames
        self._use_typecheck = use_typecheck
        self._verbose = verbose
        self._step_size = step_size

        _logger.debug(
            f"Using adjacent collector ({num_frames} frames) with required sources {self._required_capture_sources}"
        )

    @override
    def __eq__(self, other):
        return (
            isinstance(other, GroupAdjacentTime)
            and self._num_frames == other._num_frames
            and self._use_typecheck == other._use_typecheck
        )

    @override
    def __hash__(self):
        return hash((self._num_frames, self._use_typecheck))

    @override
    def __str__(self):
        return f"GroupAdjacentTime(num_frames={self._num_frames}, use_typecheck={self._use_typecheck})"

    __repr__ = __str__

    def __call__(self, mfst: data_types.Manifest, /) -> QueueGeneratorType:
        if self._use_typecheck:
            data_types.sanity.check_typeddict(mfst, data_types.Manifest)

        success = 0

        for seq, rec in mfst["sequences"].items():
            cap_list = rec["captures"]
            if len(cap_list) < self._num_frames:
                continue
            for i, cap in enumerate(cap_list[: len(cap_list) - self._num_frames + 1]):
                key = cap["primary_key"]

                caps = [cap_list[i + f]["sources"] for f in range(0, self._num_frames)]

                # Check that all required capture sources are present
                if not all(
                    isinstance(caps[n], dict)
                    and self._required_capture_sources[n].issubset(caps[n].keys())
                    for n in range(self._num_frames)
                ):
                    if self._verbose:
                        wants = list(tuple(r) for r in self._required_capture_sources)
                        has = list(tuple(c.keys()) for c in caps)
                        _logger.debug(
                            f"Skipping sequence {seq} due to missing capture sources! Wants {wants}, got {has}!"
                        )
                    continue

                item: data_types.QueueItem = {
                    "sequence": seq,
                    "frame": i,
                    "fps": 1.0,
                    "camera": rec["camera"],
                    "captures": caps,
                }

                yield key, item
                success += 1

        _logger.debug(f"Found {success} sequences with {self._num_frames} captures!")

        if success <= 0:
            raise ValueError(f"No sequences with {self._num_frames} captures found!")


class ExtractIndividualFrames(GroupAdjacentTime):
    """
    Essentially an alias to GroupAdjacentTime with num_frames=1.
    """

    def __init__(self, **kwargs):
        if "num_frames" in kwargs:
            raise ValueError(
                "num_frames is not a valid argument for ExtractIndividualImages!"
            )
        super().__init__(num_frames=1, **kwargs)
