"""
Implements the DVPQ and DSTQ metrics.

Code adapted from: https://github.com/joe-siyuan-qiao/ViP-DeepLab
"""

import itertools
import typing as T
import typing_extensions as TX
import dataclasses as D
from einops import rearrange

import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase

from unipercept.model import InputData, ModelOutput

from ._depth import DepthWriter, TRUE_DEPTH, PRED_DEPTH
from ._panoptic import PanopticWriter, PQDefinition, TRUE_PANOPTIC, PRED_PANOPTIC

FRAME_ID = "frame_id"
SEQUENCE_ID = "sequence_id"


class DVPSWriter(PanopticWriter, DepthWriter):
    """
    Writes DVPS requirements to storage.
    """

    @TX.override
    def update(self, storage: TensorDictBase, inputs: InputData, outputs: ModelOutput):
        super().update(storage, inputs, outputs)

        storage.setdefault(SEQUENCE_ID, inputs.ids[:, 0], inplace=True)
        storage.setdefault(FRAME_ID, inputs.ids[:, 1], inplace=True)


@D.dataclass(kw_only=True)
class DVPSEvaluator(DVPSWriter):
    """
    Computes (D)VPQ and (D)STQ metrics.
    """

    show_progress: bool = False
    show_summary: bool = True
    show_details: bool = False
    report_details: bool = False

    pq_definition: PQDefinition = PQDefinition.ORIGINAL

    # See Qiao et al. "ViP-DeepLab" (2020) for details on parameters
    dvpq_windows: list[int] = D.field(default_factory=lambda: [1, 2, 3, 4])
    dvpq_thresholds: list[float] = D.field(default_factory=lambda: [0.5, 0.25, 0.1])
    dstq_thresholds: list[float] = D.field(default_factory=lambda: [1.25, 1.1])

    @TX.override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        return {}

    def compute_dvpq(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, T.Any]:
        indices_per_sequence: dict[int, list[int]] = {}

        # Group by sequence
        for i, seq_id in enumerate(storage[SEQUENCE_ID]):
            indices_per_sequence.setdefault(seq_id.item(), []).append(i)

        # Sort each sequence by frame id
        for indices in indices_per_sequence.values():
            indices.sort(key=lambda i: storage.get_at(FRAME_ID, i).item())

        # Run for each window
        pq_per_win_thrs: dict[tuple[int,float], dict] = {}
        for window, threshold in itertools.product(
            self.dvpq_windows, self.dvpq_thresholds
        ):
            for indices in indices_per_sequence.values():
                pq_per_win_thrs[window, threshold] = _compute_dvpq(
                    storage, indices, window, threshold
                )

        return {} 
    
    def _compute_dvpq_at(storage: TensorDictBase, indices: list[int], window: int, threshold: float):
        """
        Computes DVPQ for a sequence of frames.
        """

        # Make groups of length `window` and compute PQ for each group
        indices = indices[: len(indices) - window + 1]
        pq_per_group = []

        for i in range(len(indices)):
            group = indices[i : i + window]

            true_seg = storage.get_at(TRUE_PANOPTIC, group).contiguous()
            pred_seg = storage.get_at(PRED_PANOPTIC, group).contiguous()
            true_dep = storage.get_at(TRUE_DEPTH, group).contiguous()
            pred_dep = storage.get_at(PRED_DEPTH, group).contiguous()

            # Mask out invalid depths
            valid_dep = true_dep > 1 & true_seg >= 0
            true_dep = true_dep[valid_dep]
            pred_dep = pred_dep[valid_dep]

            # Compute absolute relative error
            abs_rel = torch.full_like(true_seg, threshold + 1)
            abs_rel[valid_dep] = torch.abs(true_dep - pred_dep) / true_dep
            
            # Determine which pixels meet the threshold
            thres_mask = abs_rel < threshold

            pred_seg[~thres_mask] = -1

            # Stack the group into one large image
            true_seg = rearrange(true_seg, "b h w -> (b h) w")
            pred_seg = rearrange(pred_seg, "b h w -> (b h) w")

            # Compute PQ
            pq_per_group.append(
                _compute_pq(true_seg, pred_seg, self.pq_definition)
            )        

        # void_color = _get_void_color(self.object_ids, self.background_ids)
        # # device = torch.device("cpu")  # using multiprocessing

        # num_categories = len(self.object_ids) + len(self.background_ids)
        # iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        # tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        # fp = torch.zeros_like(iou)
        # fn = torch.zeros_like(fp)

        # # Loop over each sample independently: segments must not be matched across frames.
        # sample_amt = storage.batch_size[0]
        # # worker_amt = min(multiprocessing.cpu_count(), 16)
        # assert sample_amt > 0, f"Batch size must be greater than zero, got {sample_amt=}"

        # n_iter = range(sample_amt)
        # if self.show_progress:
        #     n_iter = tqdm(n_iter, desc="accumulating pqs", dynamic_ncols=True, total=sample_amt)

        # for n in n_iter:
    def compute_dstq(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, T.Any]:
        return {}

    @TX.override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}