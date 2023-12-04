"""
Path datasets are helper classes that allow to cast a local directory or paths to a dataset.
"""

import typing as T
from ._base import PerceptionDataset


def get_info():
    """Returns the info of a dataset."""
    return get_dataset(name).info

class ImagePathsDataset(PerceptionDataset):
    def __init__(self, paths: T.Sequence[tuple[str, tuple[int, int]]], ops: T.Sequence[Op]):
        self.paths = paths
        self.ops = ops

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, (sequence_id, frame_id) = self.paths[index]

        input = create_inputs(path, sequence_offset=sequence_id, frame_offset=frame_id)[0]
        for op in self.ops:
            input = op(input)
        return input