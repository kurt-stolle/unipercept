"""
Implements the DVPQ and DSTQ metrics.

Code adapted from: https://github.com/joe-siyuan-qiao/ViP-DeepLab
"""
import dataclasses as D

import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase
from typing_extensions import override

from unipercept.model import InputData, ModelOutput

from ._depth import DepthWriter
from ._panoptic import PanopticWriter

FRAME_ID = "frame_id"
SEQUENCE_ID = "sequence_id"


class DVPSWriter(PanopticWriter, DepthWriter):
    """
    Writes DVPS requirements to storage.
    """

    @override
    def update(self, storage: TensorDictBase, inputs: InputData, outputs: ModelOutput):
        super().update(storage, inputs, outputs)


@D.dataclass(kw_only=True)
class DVPSEvaluator(DVPSWriter):
    """
    Computes (D)VPQ and (D)STQ metrics.
    """

    show_progress: bool = False
    show_summary: bool = True
    show_details: bool = False

    # See Qiao et al. "ViP-DeepLab" (2020) for details on parameters
    dvpq_windows: list[int] = D.field(default_factory=lambda: [1, 2, 3, 4])
    dvpq_thresholds: list[float] = D.field(default_factory=lambda: [0.5, 0.25, 0.1])
    dstq_thresholds: list[float] = D.field(default_factory=lambda: [1.25, 1.1])

    @override
    def compute(self, storage: TensorDictBase, *, **kwargs) -> dict[str, int | float | str | bool]:
        return {}

    def compute_dvpq(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, int | float | str | bool]:
        return {}

    def compute_dstq(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, int | float | str | bool]:
        return {}

    @override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}
