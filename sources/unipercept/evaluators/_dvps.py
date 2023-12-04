"""
Implements the DVPQ and DSTQ metrics.

Code adapted from: https://github.com/joe-siyuan-qiao/ViP-DeepLab
"""
import torch
import torch.types
import dataclasses as D
from PIL import Image as pil_image
from tensordict import TensorDictBase

from unipercept.model import ModelOutput
from ._panoptic import PanopticWriter
from ._depth import DepthWriter

from typing_extensions import override


FRAME_ID = "frame_id"
SEQUENCE_ID = "sequence_id"


class DVPSWriter(PanopticWriter, DepthWriter):
    """
    Writes DVPS requirements to storage.
    """

    @override
    def update(self, storage: TensorDictBase, outputs: ModelOutput):
        super().update(storage, outputs)


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
