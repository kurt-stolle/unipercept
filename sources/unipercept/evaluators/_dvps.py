"""
Implements the DVPQ and DSTQ metrics.

Code adapted from: https://github.com/joe-siyuan-qiao/ViP-DeepLab
"""

from __future__ import annotations

import abc
import concurrent.futures
import dataclasses as D
import functools
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
    VALID_PANOPTIC,
    PanopticWriter,
    PQDefinition,
    _get_void_color,
    _panoptic_quality_update_sample,
    _preprocess_mask,
)
from unipercept.log import get_logger
from unipercept.state import check_main_process, cpus_available
from unipercept.utils.dicttools import (
    defaultdict_recurrent,
    defaultdict_recurrent_to_dict,
)

FRAME_ID = "frame_id"
SEQUENCE_ID = "sequence_id"

_logger = get_logger(__name__)


__all__ = ["DVPSWriter", "DVPSEvaluator"]


@D.dataclass(kw_only=True)
class DVPSWriter(PanopticWriter, DepthWriter):
    """
    Writes DVPS requirements to storage.
    """

    ids_key = "ids"

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unipercept import get_info

        return cls(info=get_info(name), **kwargs)

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

        storage.set(SEQUENCE_ID, sequence_id, inplace=True)  # noqa: PD002
        storage.set(FRAME_ID, frame_id, inplace=True)  # noqa: PD002


@D.dataclass(kw_only=True, slots=True)
class DVPSEvaluator(DVPSWriter):
    """
    Computes (D)VPQ and (D)STQ metrics.
    """

    show_progress: bool = True
    show_summary: bool = True
    show_details: bool = False
    report_details: bool = False

    pq_definition: PQDefinition = PQDefinition.ORIGINAL

    # See Qiao et al. "ViP-DeepLab" (2020) for details on parameters
    dvpq_windows: list[int] = D.field(default_factory=lambda: [0, 1, 2, 3, 4])
    dvpq_thresholds: list[float] = D.field(default_factory=lambda: [0.5, 0.25, 0.1])
    dstq_thresholds: list[float] = D.field(default_factory=lambda: [1.25, 1.1])

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unipercept import get_info

        return cls(info=get_info(name), **kwargs)

    @property
    def object_ids(self) -> frozenset[int]:
        return self.info.object_ids

    @property
    def background_ids(self) -> frozenset[int]:
        return self.info.background_ids

    @TX.override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        num_samples: int = storage.batch_size[0]
        indices_per_sequence: dict[int, list[int]] = {}
        for i in range(num_samples):
            valid = T.cast(torch.Tensor, storage.get_at(VALID_PANOPTIC, i, None))
            if valid is None:
                msg = f"Missing {VALID_PANOPTIC} in storage."
                raise ValueError(msg)
            if not valid.item():
                print("Skipping invalid sample")
                continue
            seq_id = T.cast(torch.Tensor, storage.get_at(SEQUENCE_ID, i))
            seq_id_item = int(seq_id.item())
            indices_per_sequence.setdefault(seq_id_item, []).append(i)

        # Sort each sequence by frame id
        for indices in indices_per_sequence.values():
            indices.sort(key=lambda i: storage.get_at(FRAME_ID, i).item())

        return {
            "vpq": self._compute_dvpq(
                storage,
                indices_per_sequence,
                self.dvpq_windows,
                None,
                **kwargs,
            ),
            "dvpq": self._compute_dvpq(
                storage,
                indices_per_sequence,
                self.dvpq_windows,
                self.dvpq_thresholds,
                **kwargs,
            ),
            "dstq": self._compute_dstq(storage, indices_per_sequence, **kwargs),
        }

    def _compute_dvpq(
        self,
        storage: TensorDictBase,
        indices_per_sequence: dict[int, list[int]],
        windows: list[int],
        thresholds: list[float] | None,
        *,
        device: torch.types.Device,
        **kwargs,
    ) -> dict[str, T.Any]:
        if len(windows) == 0:
            msg = "No windows to compute (D)VPQ."
            raise ValueError(msg)

        # Run for each window
        summaries = []
        for win, th in itertools.product(
            windows, thresholds if thresholds is not None else [0]
        ):
            with self._progress_bar(
                desc=f"DVPQ: window {win} @ {th}" if th > 0 else f"VPQ: window {win}",
                total=len(indices_per_sequence),
            ) as pbar:
                for indices in indices_per_sequence.values():
                    pbar.update(1)
                    if len(indices) == 0:
                        raise ValueError("Empty sequence.")

                    if self.pq_definition & PQDefinition.ORIGINAL:
                        sum = self._compute_dvpq_at(
                            storage,
                            indices,
                            window=win,
                            threshold=th,
                            device=device,
                            allow_stuff_instances=True,
                        )
                        sum["definition"] = "original"
                        summaries.append(sum)
                    if self.pq_definition & PQDefinition.BALANCED:
                        sum = self._compute_dvpq_at(
                            storage,
                            indices,
                            window=win,
                            threshold=th,
                            device=device,
                            allow_stuff_instances=False,
                        )
                        sum["definition"] = "balanced"
                        summaries.append(sum)

        # Combine summarie
        df = pd.concat(summaries, ignore_index=True)
        result = defaultdict_recurrent()
        supercats = ["all", "thing", "stuff"]

        for definition, df_d in df.groupby("definition"):
            for win, df_w in df_d.groupby("window"):
                win_key = _format_summary_number(int(win), "w")
                # Compute mean over all thresholds
                win_mean = result[definition][win_key]["t_mean"] = {
                    c: {
                        metric: df_m[c].mean()
                        for metric, df_m in df_w.groupby("metric")
                    }
                    for c in supercats
                }

                if thresholds is not None:
                    # Compute at each threshold
                    for th, df_t in df_w.groupby("threshold"):
                        th_key = _format_summary_number(float(th), "t")
                        result[definition][win_key][th_key] = {
                            c: {
                                metric: df_m[c].mean()
                                for metric, df_m in df_t.groupby("metric")
                            }
                            for c in supercats
                        }

                    result[definition][win_key]["t_all"] = win_mean
                else:
                    result[definition][win_key] = win_mean

            # Compute mean over all windows
            if thresholds is not None:
                for th, df_t in df_d.groupby("threshold"):
                    result[definition]["w_all"][
                        _format_summary_number(int(th), "t")
                    ] = {
                        c: {
                            metric: df_m[c].mean()
                            for metric, df_m in df_t.groupby("metric")
                        }
                        for c in supercats
                    }

        for definition, df_d in df.groupby("definition"):
            # Compute mean over all windows and thresholds
            for metric, df_m in df_d.groupby("metric"):
                for c in supercats:
                    c_metric = df_m[c].mean()
                    result[definition]["overall"][c][metric] = c_metric
        return defaultdict_recurrent_to_dict(result)

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

        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)

        # Loop over each sample independently: segments must not be matched across frames.
        compute_dvpq_at_group = functools.partial(
            _compute_dvpq_at_group,
            storage=storage,
            device=device,
            threshold=threshold,
            void_color=void_color,
            allow_stuff_instances=allow_stuff_instances,
            allow_unknown_category=allow_unknown_category,
            num_categories=num_categories,
            object_ids=self.object_ids,
            background_ids=self.background_ids,
        )
        sample_amt = len(indices)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(cpus_available(), 8)
        ) as pool:
            for result in pool.map(
                compute_dvpq_at_group,
                (indices[i : i + window] for i in range(sample_amt)),
            ):
                iou += result[0]
                tp += result[1]
                fp += result[2]
                fn += result[3]
        # for i in range(sample_amt):
        #     group = indices[i : i + window]
        #     result = compute_dvpq_at_group(group)

        #     iou += result[0]
        #     tp += result[1]
        #     fp += result[2]
        #     fn += result[3]

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
            ("all", tn_mask),
            ("thing", tn_mask & th_mask),
            ("stuff", tn_mask & st_mask),
        ]:
            # n_masked = n_valid[mask].sum().item()
            summary[name] = {
                "PQ": pq[mask].mean().item(),
                "SQ": rq[mask].mean().item(),
                "RQ": fp[mask].mean().item(),
                # "IoU": iou[mask].mean().item(),
                # "TP": tp[mask].sum().item() / n_masked,
                # "FP": fp[mask].sum().item() / n_masked,
                # "FN": fn[mask].sum().item() / n_masked,
            }

        summary_df = self._tabulate(summary)
        summary_df["window"] = window
        summary_df["threshold"] = threshold

        return summary_df

    def _compute_dstq(
        self,
        storage: TensorDictBase,
        indices_per_sequence: dict[int, list[int]],
        *,
        device: torch.types.Device,
        **kwargs,
    ) -> dict[str, T.Any]:
        if len(self.dstq_thresholds) == 0:
            return {}
        return {}

    @TX.override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}

    def _tabulate(self, result: dict[str, dict[str, float]]) -> pd.DataFrame:
        data: dict[str, list[float]] = {}
        groups = []

        for group_name, metrics in result.items():
            groups.append(group_name)
            for metric_name, metric_value in metrics.items():
                data[metric_name] = data.get(metric_name, []) + [metric_value]

        data_list = []
        for key, values in data.items():
            data_list.append([key] + values)

        df = pd.DataFrame(
            data_list,
            columns=["metric"] + groups,
        )

        return df


def _format_summary_number(num: T.Any, name: str) -> str:
    res = str(num)
    if "." in res:
        return res.replace(".", name)
    return res + name


def _compute_dvpq_at_group(
    group: list[int],
    *,
    storage,
    device,
    threshold: int,
    void_color,
    allow_stuff_instances,
    allow_unknown_category,
    num_categories,
    object_ids,
    background_ids,
):
    true_seg = storage.get_at(TRUE_PANOPTIC, group).to(device, non_blocking=True)
    pred_seg = storage.get_at(PRED_PANOPTIC, group).to(device, non_blocking=True)
    true_dep = storage.get_at(TRUE_DEPTH, group).to(device, non_blocking=True)
    pred_dep = storage.get_at(PRED_DEPTH, group).to(device, non_blocking=True)
    assert pred_dep.shape == true_dep.shape
    assert pred_seg.shape == true_seg.shape
    assert true_dep.dtype == torch.float32
    assert pred_dep.dtype == torch.float32

    # Mask out invalid depths
    if threshold > 0:
        valid_dep = true_dep > 0
        valid_dep = torch.where(true_seg >= 0, valid_dep, False)
        true_dep = true_dep[valid_dep]
        pred_dep = pred_dep[valid_dep]

        # Compute absolute relative error
        abs_rel = torch.full_like(pred_seg, threshold + 1, dtype=pred_dep.dtype)
        abs_rel[valid_dep] = torch.abs(true_dep - pred_dep) / true_dep

        # Determine which pixels meet the threshold
        thres_mask = abs_rel < threshold

        pred_seg[~thres_mask].fill_(-1)

    # Stack the group into one large image
    true_seg = rearrange(true_seg, "b h w -> (b h) w")
    pred_seg = rearrange(pred_seg, "b h w -> (b h) w")

    # Compute PQ
    pred_seg = _preprocess_mask(
        object_ids,
        background_ids,
        pred_seg,
        void_color=void_color,
        allow_unknown_category=allow_unknown_category,
    )
    true_seg = _preprocess_mask(
        object_ids,
        background_ids,
        true_seg,
        void_color=void_color,
        allow_unknown_category=True,
    )

    return _panoptic_quality_update_sample(
        pred_seg,
        true_seg,
        void_color=void_color,
        background_ids=(background_ids if not allow_stuff_instances else None),
        num_categories=num_categories,
    )
