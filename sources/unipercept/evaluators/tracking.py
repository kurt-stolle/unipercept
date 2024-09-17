r"""
Tracking evaluators for object tracking tasks, e.g. HOTA, MOTA, IDF1, etc.
"""

from __future__ import annotations

import dataclasses as D
import typing as T

import typing_extensions as TX

from unipercept.log import get_logger

from ._base import Evaluator

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
    def _update(
        self,
        storage: TensorDictBase,
        inputs: InputData,
        outputs: TensorDictBase,
        **kwargs,
    ):
        super()._update(storage, inputs, outputs, **kwargs)

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
    def _compute(self, *args, **kwargs):
        return super()._compute(*args, **kwargs)

    @TX.override
    def _plot(self, *args, **kwargs):
        return super()._plot(*args, **kwargs)


class CLEARTrackingEvaluator(VideoIDWriter):
    r"""
    Evaluates multiple object tracking using the CLEAR [1] metrics.

    References
    ----------
    [1] Bernardin et al., `Evaluating multiple object tracking performance: the CLEAR MOT metrics <https://dl.acm.org/doi/pdf/10.1155/2008/246309>`_ (2008)
    """

    @TX.override
    def _compute(
        self,
        storage: TensorDictBase,
        inputs: InputData,
        outputs: TensorDictBase,
        **kwargs,
    ):
        result = super()._compute(storage, inputs, outputs, **kwargs)

        return result


class HOTATrackingEvaluator(VideoIDWriter):
    r"""
    Evaluates multiple object tracking using the HOTA [1] metrics.

    See Also
    --------
    - Official `reference implementation <https://github.com/nekorobov/HOTA-metrics>`_ by the auhors of [1].

    References
    ----------
    [1] Luiten et al., `HOTA: A Higher Order Metric for Evaluating Multi-object Tracking <https://link.springer.com/article/10.1007/s11263-020-01375-2>`_ (2020)
    """

    @TX.override
    def _compute(
        self,
        storage: TensorDictBase,
        inputs: InputData,
        outputs: TensorDictBase,
        **kwargs,
    ):
        result = super()._compute(storage, inputs, outputs, **kwargs)

        return result
