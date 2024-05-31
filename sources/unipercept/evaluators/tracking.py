r"""
Tracking evaluators for object tracking tasks, e.g. HOTA, MOTA, IDF1, etc.
"""

from __future__ import annotations

import dataclasses as D
import typing as T

import typing_extensions as TX
from PIL import Image as pil_image

from unipercept.evaluators import Evaluator
from unipercept.log import get_logger

if T.TYPE_CHECKING:
    from tensordict import TensorDictBase

    from unipercept.model import InputData

_logger = get_logger(__name__)


@D.dataclass(kw_only=True)
class VideoIDWriter(Evaluator):
    """
    Writes DVPS requirements to storage.
    """

    @property
    def key_video_sequence(self) -> str:
        return self._get_storage_key("sequence", "video")

    @property
    def key_video_frame(self) -> str:
        return self._get_storage_key("frame", "video")

    @TX.override
    def update(
        self, storage: TensorDictBase, inputs: InputData, outputs: TensorDictBase
    ):
        super().update(storage, inputs, outputs)

        target_keys = {self.key_video_sequence, self.key_video_frame}
        storage_keys = storage.keys(include_nested=True, leaves_only=True)
        assert storage_keys is not None
        if target_keys.issubset(storage_keys):
            return

        sequence_id, frame_id = inputs.ids.unbind(-1)
        for key, item in {
            self.key_video_sequence: sequence_id,
            self.key_video_frame: frame_id,
        }.items():
            storage.set(key, item, inplace=True)

    @TX.override
    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs)

    @TX.override
    def plot(self, *args, **kwargs):
        return super().plot(*args, **kwargs)
