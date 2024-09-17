"""This module contains methods that can be used to generate a queue from a manifest."""

from __future__ import annotations

import typing as T
from typing import override

from unipercept.log import logger

if T.TYPE_CHECKING:
    from unipercept.data.sets import (
        CaptureSourceKey,
        Manifest,
        QueueGenerator,
        QueueItem,
    )


class GroupAdjacentTime:
    """
    Queue that collects queue items of a specified length from the manifest, where items
    in the queue are temporally adjacent to each other.

    This is useful for tasks such as optical flow, where the input is a sequence of images.
    """

    def __init__(
        self,
        num_frames: int,
        *,
        required_capture_sources: (
            set[CaptureSourceKey] | T.Sequence[set[CaptureSourceKey]] | None
        ) = None,
        verbose: bool = False,
        step_size: T.Collection[int] | None = None,
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
                "Expected a sequence of strings or a sequence of sequences of strings!"
            )

        self._num_frames = num_frames
        self._verbose = verbose
        self._step_size = set(step_size) if step_size is not None else {1}
        # logger.debug(
        #     f"Using adjacent collector ({num_frames} frames) with required sources {self._required_capture_sources}"
        # )

    @override
    def __str__(self):
        repr_args = [str(self._num_frames)]
        repr_kwargs = {
            "step_size": self._step_size,
            "sources": str(self._required_capture_sources),
        }
        args = ", ".join(repr_args + [f"{k}={str(v)}" for k, v in repr_kwargs.items()])
        return f"GroupAdjacentTime({args})"

    __repr__ = __str__

    def __call__(self, mfst: Manifest, /) -> QueueGenerator:
        success = 0

        for seq, rec in mfst["sequences"].items():
            cap_list = rec["captures"]

            # Check that the sequence has enough captures
            if len(cap_list) < self._num_frames:
                continue

            # For every starting frame, create a list of other frames that are within the max_distance
            indices_from_start = [
                (i, [i + int(j * s) for j in range(1, self._num_frames)])
                for s in self._step_size
                for i in range(len(cap_list))
            ]
            # Filter out indices that are out of bounds
            indices_from_start = [
                (i, [j for j in js if 0 <= j < len(cap_list)])
                for i, js in indices_from_start
            ]
            # Filter out invalid lengths
            indices_from_start = set(
                (i, tuple(js))
                for i, js in indices_from_start
                if len(js) == self._num_frames - 1
            )
            indices_from_start = sorted(indices_from_start, key=lambda x: (x[0], *x[1]))
            assert all(
                len(v) == self._num_frames - 1 for _, v in indices_from_start
            ), indices_from_start
            # Collect the sequence for each frame and the designated number of subsequent frames
            for i, indices in indices_from_start:
                cap_start = cap_list[i]
                sources = [cap_start["sources"]] + [
                    cap_list[j]["sources"] for j in indices
                ]

                # Check that all required capture sources are present
                if not all(
                    isinstance(sources[n], dict)
                    and self._required_capture_sources[n].issubset(sources[n].keys())
                    for n in range(self._num_frames)
                ):
                    if self._verbose:
                        wants = list(tuple(r) for r in self._required_capture_sources)
                        has = list(tuple(c.keys()) for c in sources)
                        logger.debug(
                            f"Skipping sequence {seq} due to missing capture sources! Wants {wants}, got {has}!"
                        )
                    continue

                item: QueueItem = {
                    "sequence": seq,
                    "frame": i,
                    "fps": 1.0,
                    "camera": rec["camera"],  # type: ignore
                    "captures": sources,
                }
                primary_key = "+".join([cap_start["primary_key"], *map(str, indices)])
                yield primary_key, item
                success += 1
        if success <= 0:
            raise ValueError(f"No sequences with {self._num_frames} captures found!")
        # logger.debug(f"{self}: {success} pairs with {self._num_frames} captures!")


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
