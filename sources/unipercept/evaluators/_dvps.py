"""
Implements the DVPQ and DSTQ metrics.

Code adapted from: https://github.com/joe-siyuan-qiao/ViP-DeepLab
"""
from __future__ import annotations

import dataclasses as D
import itertools
import typing as T

import pandas as pd
import torch
import torch.multiprocessing as M
import torch.types
import typing_extensions as TX
from einops import rearrange
from PIL import Image as pil_image
from tensordict import TensorDictBase
from tqdm import tqdm

import unipercept.evaluators.helpers as H
from unipercept.evaluators._depth import PRED_DEPTH, TRUE_DEPTH, DepthWriter
from unipercept.evaluators._panoptic import (
    PRED_PANOPTIC,
    TRUE_PANOPTIC,
    PanopticWriter,
    PQDefinition,
    _get_void_color,
    _panoptic_quality_update_sample,
    _preprocess_mask,
)
from unipercept.log import get_logger
from unipercept.state import check_main_process

FRAME_ID = "frame_id"
SEQUENCE_ID = "sequence_id"

_logger = get_logger(__name__)


__all__ = ["DVPSWriter", "DVPSEvaluator"]


class DVPSWriter(PanopticWriter, DepthWriter):
    """
    Writes DVPS requirements to storage.
    """

    ids_key = "ids"

    @TX.override
    def update(
        self, storage: TensorDictBase, inputs: TensorDictBase, outputs: TensorDictBase
    ):
        super().update(storage, inputs, outputs)

        storage_keys = storage.keys(include_nested=True, leaves_only=True)
        if SEQUENCE_ID in storage_keys and FRAME_ID in storage_keys:
            return

        combined_id = inputs.get(self.ids_key)
        assert (
            combined_id.shape[-1] == 2
        ), f"Expected {self.ids_key} to have shape (..., 2). Got {combined_id.shape}."

        sequence_id = combined_id[..., 0]
        frame_id = combined_id[..., 1]

        storage.set(SEQUENCE_ID, sequence_id, inplace=True)
        storage.set(FRAME_ID, frame_id, inplace=True)


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

    @classmethod
    @TX.override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @property
    def object_ids(self) -> frozenset[int]:
        return self.info.object_ids

    @property
    def background_ids(self) -> frozenset[int]:
        return self.info.background_ids

    @TX.override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        return {
            "dvpq": self.compute_dvpq(storage, **kwargs),
            "dstq": self.compute_dstq(storage, **kwargs),
        }

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
        summaries = []
        for window, threshold in itertools.product(
            self.dvpq_windows, self.dvpq_thresholds
        ):
            for indices in indices_per_sequence.values():
                if self.pq_definition & PQDefinition.ORIGINAL:
                    sum = self._compute_dvpq_at(
                        storage,
                        indices,
                        window,
                        threshold,
                        **kwargs,
                        allow_stuff_instances=True,
                    )
                    sum.Definition = "original"
                    summaries.append(sum)
                if self.pq_definition & PQDefinition.BALANCED:
                    sum = self._compute_dvpq_at(
                        storage,
                        indices,
                        window,
                        threshold,
                        **kwargs,
                        allow_stuff_instances=False,
                    )
                    sum.Definition = "balanced"
                    summaries.append(sum)

        # Combine summaries
        df = pd.concat(summaries, ignore_index=True)

        # Create results dict, which should have:
        #   PQ, SQ, RQ, IoU, TP, FP, FN for All, Thing and Stuff
        # grouped by window and threshold, i.e:
        #   results[definition][window][threshold][group][metric]
        # overall as
        #   results[definition]["overall"][group][metric]

        result = {}

        for definition, df_d in df.groupby("Definition"):
            result[definition] = {}
            for window, df_w in df_d.groupby("Window"):
                window_key = "w_" + str(window).replace(".", "_")
                result[definition][window_key] = {}
                for threshold, df_t in df_w.groupby("Threshold"):
                    threshold_key = "t_" + str(threshold).replace(".", "_")
                    result[definition][window_key][threshold_key] = {}
                    for metric, df_m in df_t.groupby("Metric"):
                        result[definition][window_key][threshold_key]["all"][
                            metric
                        ] = df_m["All"].mean()
                        result[definition][window_key][threshold_key]["thing"][
                            metric
                        ] = df_m["Thing"].mean()
                        result[definition][window_key][threshold_key]["stuff"][
                            metric
                        ] = df_m["Stuff"].mean()

        for definition, df_d in df.groupby("Definition"):
            result[definition]["mean"] = {}
            for metric, df_m in df_d.groupby("Metric"):
                result[definition]["mean"]["all"][metric] = df_m["All"].mean()
                result[definition]["mean"]["thing"][metric] = df_m["Thing"].mean()
                result[definition]["mean"]["stuff"][metric] = df_m["Stuff"].mean()

        return result

    def _compute_dvpq_at(
        self,
        storage: TensorDictBase,
        indices: list[int],
        window: int,
        threshold: float,
        *,
        device: torch.types.Device,
        allow_stuff_instances: bool,
        allow_unknown_category=False,
    ) -> pd.DataFrame:
        """
        Computes DVPQ for a sequence of frames.
        """

        # Make groups of length `window` and compute PQ for each group
        indices = indices[: len(indices) - window + 1]

        void_color = _get_void_color(self.object_ids, self.background_ids)
        # device = torch.device("cpu")  # using multiprocessing

        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)

        # Loop over each sample independently: segments must not be matched across frames.
        sample_amt = len(indices)
        n_iter = range(sample_amt)
        if self.show_progress:
            n_iter = tqdm(
                n_iter,
                desc="accumulating pqs",
                dynamic_ncols=True,
                total=sample_amt,
                disable=not check_main_process(local=True),
            )

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
            pred_seg = _preprocess_mask(
                self.object_ids,
                self.background_ids,
                pred_seg,
                void_color=void_color,
                allow_unknown_category=allow_unknown_category,
            )
            true_seg = _preprocess_mask(
                self.object_ids,
                self.background_ids,
                true_seg,
                void_color=void_color,
                allow_unknown_category=True,
            )

            result = _panoptic_quality_update_sample(
                pred_seg,
                true_seg,
                void_color=void_color,
                background_ids=self.background_ids
                if not allow_stuff_instances
                else None,
                num_categories=num_categories,
            )

            iou += result[0]
            tp += result[1]
            fp += result[2]
            fn += result[3]

        # Compute PQ = SQ * RQ
        sq = H.stable_divide(iou, tp)
        rq = H.stable_divide(tp, tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq

        # Convert to percentages
        sq *= 100
        rq *= 100
        pq *= 100

        # Total valid values
        n_valid = tp + fp + fn

        # Summarizing values
        summary = {}

        # Mask out categories that have only true negatives
        tn_mask: torch.Tensor = n_valid > 0
        th_mask: torch.Tensor = H.isin(
            torch.arange(num_categories, device=device), list(self.object_ids)
        )
        st_mask: torch.Tensor = H.isin(
            torch.arange(num_categories, device=device), list(self.background_ids)
        )

        for name, mask in [
            ("All", tn_mask),
            ("Thing", tn_mask & th_mask),
            ("Stuff", tn_mask & st_mask),
        ]:
            n_masked = n_valid[mask].sum().item()
            summary[name] = {
                "PQ": pq[mask].mean().item(),
                "SQ": rq[mask].mean().item(),
                "RQ": fp[mask].mean().item(),
                "IoU": iou[mask].mean().item(),
                "TP": tp[mask].sum().item() / n_masked,
                "FP": fp[mask].sum().item() / n_masked,
                "FN": fn[mask].sum().item() / n_masked,
            }
        summary_df = self._tabulate(summary)
        summary_df.Window = window
        summary_df.Threshold = threshold

        return summary_df

    def compute_dstq(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, T.Any]:
        return {}

    @TX.override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}

    def _tabulate(self, result: dict[str, dict[str, float]]) -> pd.DataFrame:
        data: dict[str, list[float]] = {}
        groups = []

        for group_name, metrics in result.items():
            groups.append(group_name.capitalize())
            for metric_name, metric_value in metrics.items():
                data[metric_name] = data.get(metric_name, []) + [metric_value]

        data_list = []
        for key, values in data.items():
            data_list.append([key] + values)

        df = pd.DataFrame(
            data_list,
            columns=["Metric"] + groups,
        )

        return df
