"""
Implements the DVPQ and DSTQ metrics.

See Also
--------
- Reference implementation <https://github.com/joe-siyuan-qiao/ViP-DeepLab>
"""

from __future__ import annotations

import concurrent.futures
import dataclasses as D
import enum as E
import functools
import functools as F
import itertools
import typing as T
import warnings

import pandas as pd
import regex as re
import torch
import torch.types
import typing_extensions as TX
from einops import rearrange
from tensordict import TensorDictBase

import unipercept.data.sets as datasets
import unipercept.vision.point
from unipercept import file_io, state
from unipercept.data import tensors
from unipercept.evaluators import depth, segmentation, tracking

from ._base import Evaluator
from ._common import isin, stable_divide
from ._export import ExportFormat, export_dataframe

# -------------------------- #
# ViP-DeepLab reference mode #
# -------------------------- #


class ViPDeepLabMode(E.StrEnum):
    SEMKITTI = E.auto()
    CITYSCAPES = E.auto()


@D.dataclass(kw_only=True)
class ViPDeepLabWriter(Evaluator):
    r"""
    Export DVPS results such that they can be evaluated using the reference
    implementation of ViP-DeepLab.

    See Also
    --------

    - `Cityscapes Reference <https://github.com/joe-siyuan-qiao/ViP-DeepLab/blob/master/cityscapes-dvps/eval_dvpq.py>`_
    - `SemKITTI References <https://github.com/joe-siyuan-qiao/ViP-DeepLab/tree/master/semkitti-dvps>`_
    """

    evaluation_mode: ViPDeepLabMode = ViPDeepLabMode.CITYSCAPES
    sequence_pattern: re.Pattern = D.field(
        default_factory=F.partial(re.compile, r"(?P<origin>.+)/(?P<sequence>\d+)$"),
        metadata={
            "help": (
                "Regular expression pattern to extract intra-origin sequence "
                "information from the origin name, which is used to create the "
                "output path. E.g. sequence `cityscapes/dvps/0001` will have "
                "origin `cityscapes/dvps` and sequence `0001`."
            )
        },
    )
    export_name: str = "vip-export"
    file_format = "{sequence:06d}_{frame:06d}"
    drop_origin = True

    @TX.override
    def _update(self, *args, **kwargs):
        super()._update(*args, **kwargs)
        self._update_write_export(*args, **kwargs)

    def _update_write_export(
        self,
        storage: TensorDictBase,
        inputs: TensorDictBase,
        outputs: TensorDictBase,
        *,
        path: file_io.Path,
        sources: list[datasets.QueueItem] | None = None,
        **kwargs,
    ) -> None:
        for src, inp, out in zip(sources, inputs, outputs, strict=True):
            seq_name = src["sequence"]
            seq_match = self.sequence_pattern.match(seq_name)
            if seq_match is None:
                msg = f"{seq_name!r} does not match {self.sequence_pattern.pattern!r}!"
                raise ValueError(msg)
            origin, seq_id = seq_match.group("origin"), seq_match.group("sequence")
            try:
                seq_id = int(seq_id)
            except ValueError:
                # msg = f"Sequence ID {seq_id!r} is not an integer!"
                # raise ValueError(msg)
                pass

            file_base = self.file_format.format(sequence=seq_id, frame=src["frame"])
            root = path / self.export_name
            if not self.drop_origin:
                root = root / origin
            for subdir in ("video_sequence/val", "pred_panoptic", "pred_depth"):
                (root / subdir).mkdir(parents=True, exist_ok=True)

            match self.evaluation_mode:
                case ViPDeepLabMode.SEMKITTI:
                    msg = "SemKITTI is not supported yet."
                    raise NotImplementedError(msg)
                case ViPDeepLabMode.CITYSCAPES:
                    tensors.save_panoptic(
                        inp.get_at(
                            ("captures", "segmentations"), self.pair_index
                        ).squeeze(),
                        root / "video_sequence" / "val" / f"{file_base}_gtFine.png",
                        format=tensors.LabelsFormat.PNG_UINT16,
                        void_id=32,
                        max_ins=1000,
                    )
                    tensors.save_panoptic(
                        out["panoptic_segmentation"].squeeze(),
                        root / "pred_panoptic" / f"{file_base}.png",
                        format=tensors.LabelsFormat.CITYSCAPES_DVPS,
                    )

            _save_dep = F.partial(
                tensors.DepthMap.save, format=tensors.DepthFormat.DEPTH_INT16
            )
            _save_dep(
                inp.get_at(("captures", "depths"), self.pair_index),
                root / "video_sequence" / "val" / f"{file_base}_depth.png",
            )
            _save_dep(
                out["depth"].squeeze(),
                root / "pred_depth" / f"{file_base}.png",
            )

    @TX.override
    def _compute(
        self, storage: TensorDictBase, return_dataframe: bool = False, **kwargs
    ) -> dict[str, T.Any]:
        return {}  # run manually


# ---------------------------- #
# Canonical evaluator for DVPS #
# ---------------------------- #


class DVPSMetric(E.StrEnum):
    """
    Enumeration of (D)VPS metrics.
    """

    VPQ = E.auto()
    STQ = E.auto()
    DVPQ = E.auto()
    DSTQ = E.auto()


DEFAULT_DVPS_METRICS = frozenset((DVPSMetric.DVPQ, DVPSMetric.VPQ))


@D.dataclass(kw_only=True)
class DVPSEvaluator(
    tracking.VideoIDWriter, segmentation.SegmentationWriter, depth.DepthWriter
):
    """
    Computes (D)VPQ and (D)STQ metrics.

    Because these metrics are largely the same, the implementation is combined into one evaluator.
    Users can set the ``dvps_metrics`` attribute to specify which metrics to compute specifically.
    """

    report_details: bool = False

    pq_definition: segmentation.PQDefinition = segmentation.PQDefinition.ORIGINAL

    dvps_metrics: T.Collection[DVPSMetric] = D.field(
        default=DEFAULT_DVPS_METRICS,
        metadata={
            "help": f"The DVPS metrics to compute. Default: {DEFAULT_DVPS_METRICS}.",
        },
    )

    # See Qiao et al. "ViP-DeepLab" (2020) for details on parameters
    dvpq_windows: list[int] = D.field(default_factory=lambda: [1, 2, 3, 4])
    dvpq_thresholds: list[float] = D.field(default_factory=lambda: [0.5, 0.25, 0.1])
    dstq_thresholds: list[float] = D.field(default_factory=lambda: [1.25])
    dvpq_export: list[str] = D.field(
        default_factory=lambda: [ExportFormat.LATEX, ExportFormat.CSV]
    )

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        self.dvps_metrics = frozenset(self.dvps_metrics)
        self.dvpq_windows = list(map(int, self.dvpq_windows))
        self.dvpq_thresholds = list(map(float, self.dvpq_thresholds))
        self.dstq_thresholds = list(map(float, self.dstq_thresholds))

    @property
    def object_ids(self) -> frozenset[int]:
        return self.info.object_ids

    @property
    def background_ids(self) -> frozenset[int]:
        return self.info.background_ids

    @TX.override
    def _compute(
        self,
        storage: TensorDictBase,
        *,
        return_dataframe: bool = False,
        path: file_io.Path,
        **kwargs,
    ) -> dict[str, T.Any]:
        result = super()._compute(storage, path=path, **kwargs)

        if len(self.dvps_metrics) == 0:
            msg = "No DVPS metrics were specified. Skipping computation."
            warnings.warn(msg, stacklevel=2)
            return result

        num_samples: int = len(storage)
        indices_per_sequence: dict[int, list[int]] = {}
        for i in range(num_samples):
            valid = T.cast(
                torch.Tensor, storage.get_at(self.segmentation_key_valid, i, None)
            )
            if valid is None:
                msg = f"Missing {self.segmentation_key_valid} in storage."
                raise ValueError(msg)
            if not valid:
                self.logger.warning("Skipping invalid sample at index %d!", i)
                continue
            seq_id = T.cast(torch.Tensor, storage.get_at(self.key_video_sequence, i))
            seq_id_item = int(seq_id.item())
            indices_per_sequence.setdefault(seq_id_item, []).append(i)

        # Sort each sequence by frame id
        for indices in indices_per_sequence.values():
            indices.sort(
                key=lambda i: storage.get_at(self.segmentation_key_valid, i).item()
            )

        if DVPSMetric.VPQ in self.dvps_metrics:
            assert DVPSMetric.VPQ not in result, result.keys()
            result[DVPSMetric.VPQ] = self._compute_dvpq(
                storage,
                indices_per_sequence,
                self.dvpq_windows,
                None,
                return_dataframe=return_dataframe,
                path=path / "vpq",
                **kwargs,
            )
        if DVPSMetric.DVPQ in self.dvps_metrics:
            assert DVPSMetric.DVPQ.value not in result, result.keys()
            result[DVPSMetric.DVPQ.value] = self._compute_dvpq(
                storage,
                indices_per_sequence,
                self.dvpq_windows,
                self.dvpq_thresholds,
                path=path / "dvpq",
                return_dataframe=return_dataframe,
                **kwargs,
            )

        if DVPSMetric.STQ in self.dvps_metrics:
            assert DVPSMetric.STQ.value not in result, result.keys()
            result[DVPSMetric.STQ.value] = self._compute_dstq(
                storage,
                indices_per_sequence,
                return_dataframe=return_dataframe,
                **kwargs,
            )

        return result

    def _compute_dvpq(
        self,
        storage: TensorDictBase,
        indices_per_sequence: dict[int, list[int]],
        windows: list[int],
        thresholds: list[float] | None,
        *,
        device: torch.types.Device,
        return_dataframe: bool = False,
        path: file_io.Path | None = None,
        **kwargs,
    ) -> dict[str, T.Any]:
        if len(windows) == 0:
            msg = "No windows to compute (D)VPQ."
            raise ValueError(msg)

        metric_name = "DVPQ" if thresholds is not None else "VPQ"

        # Run for each window
        summaries: list[tuple[str, str, float, float, float]] = []
        thresholds_values = thresholds if thresholds is not None else [0]
        win_th_prod = list(itertools.product(windows, thresholds_values))
        work = list(range(len(win_th_prod)))

        self.logger.debug(
            "Starting evaluation tasks for window/threshold pairs: %s",
            ", ".join(map(str, win_th_prod)),
        )

        with (
            self._progress_bar(
                desc=metric_name, total=len(indices_per_sequence) * len(win_th_prod)
            ) as pbar,
            state.split_between_processes(work) as work_split,
        ):
            for work_index in work_split:
                win_mean, th = win_th_prod[work_index]
                if th > 0:
                    pbar.set_postfix({"threshold": th, "window": win_mean})
                else:
                    pbar.set_postfix({"window": win_mean})
                for indices in indices_per_sequence.values():
                    pbar.update(1)
                    if len(indices) == 0:
                        raise ValueError("Empty sequence.")

                    sum = self._compute_dvpq_at(
                        storage,
                        indices,
                        window=win_mean,
                        threshold=th,
                        device=device,
                        allow_stuff_instances=True,
                    )
                    summaries.append(sum)

        # Synchronize
        state.barrier()

        # Combine summaries
        summaries = state.gather_object(summaries)

        if not state.check_main_process():
            return {}

        vars = ["pq_all", "pq_thing", "pq_stuff"]
        idxs = ["window", "threshold"]
        df = pd.DataFrame.from_records(summaries, columns=[*idxs, *vars])
        df = df.groupby(idxs).mean().reset_index()

        df_merge = pd.pivot_table(df, values=vars, index=idxs).reset_index()

        mean_rows = []
        for idx in idxs:
            idx_mean = df.groupby(idx)[vars].mean().reset_index()
            idx_mean[[n for n in idxs if n != idx]] = "mean"
            mean_rows.append(idx_mean)

        all_mean = df_merge[vars].mean().to_frame().T
        for idx in idxs:
            all_mean[idx] = "mean"
        mean_rows.append(all_mean)

        df_merge = pd.concat([df, *mean_rows], ignore_index=True)
        df_merge = df.sort_values(by=["window", "threshold"])

        if path is not None and len(self.dvpq_export) > 0:
            for export_format in self.dvpq_export:
                export_path = path / "summary"
                export_path = export_path.with_suffix(export_format)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Exporting {metric_name} to {export_path}")
                export_dataframe(df, export_path, format=export_format)

        if self.show_summary:
            self._show_table(f"{metric_name} metrics", df_merge)

        if return_dataframe:
            return df_merge

        return {
            f"{metric_name}_all": df["pq_all"].mean(),
            f"{metric_name}_thing": df["pq_thing"].mean(),
            f"{metric_name}_stuff": df["pq_stuff"].mean(),
        }

    def _compute_dvpq_at(
        self,
        storage: TensorDictBase,
        indices: list[int],
        window: int,
        threshold: float,
        *,
        device: torch.types.Device,
        allow_stuff_instances: bool,
        allow_unknown_category=True,
    ) -> tuple[str, str, float, float, float]:
        """
        Computes DVPQ for a sequence of frames.
        """

        # Make groups of length `window` and compute PQ for each group
        void_color = segmentation.find_void_color(self.object_ids, self.background_ids)

        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)
        abs_rel = torch.tensor(0, device=device, dtype=torch.double)

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
            key_segmentation_true=self.segmentation_key_true,
            key_segmentation_pred=self.segmentation_key_pred,
            key_depth_true=self.depth_key_true,
            key_depth_pred=self.depth_key_pred,
        )
        sample_amt = len(indices)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
        ) as pool:
            for result in pool.map(
                compute_dvpq_at_group,
                (indices[i : i + window] for i in range(sample_amt - window + 1)),
            ):
                iou += result[0][:-1]
                tp += result[1][:-1]
                fp += result[2][:-1]
                fn += result[3][:-1]
                abs_rel += result[4]
        abs_rel /= sample_amt - window + 1

        # Compute PQ = SQ * RQ
        sq = stable_divide(iou, tp)
        rq = stable_divide(tp, tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq

        # Total valid values
        n_valid = tp + fp + fn

        # Mask out categories that have only true negatives
        tn_mask: torch.Tensor = n_valid > 0
        th_mask: torch.Tensor = isin(
            torch.arange(num_categories, device=device), list(self.object_ids)
        )
        st_mask: torch.Tensor = isin(
            torch.arange(num_categories, device=device), list(self.background_ids)
        )

        if not tn_mask.any():
            self.logger.warning("No valid categories found! Writing zeros for (D)VPQ.")

        # Compute final per window/threshold pair PQ values
        pq_all = pq[tn_mask].mean().nan_to_num().item() * 100
        pq_thing = pq[tn_mask & th_mask].mean().nan_to_num().item() * 100
        pq_stuff = pq[tn_mask & st_mask].mean().nan_to_num().item() * 100

        return (
            _format_summary_number(int(window), "w"),
            _format_summary_number(float(threshold), "t"),
            pq_all,
            pq_thing,
            pq_stuff,
        )

    def _compute_dstq(
        self,
        storage: TensorDictBase,
        indices_per_sequence: dict[int, list[int]],
        *,
        device: torch.types.Device,
        return_dataframe: bool = False,
        **kwargs,
    ) -> dict[str, T.Any]:
        if len(self.dstq_thresholds) == 0:
            return {} if not return_dataframe else pd.DataFrame()
        return {} if not return_dataframe else pd.DataFrame()

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


def _compute_dvpq_at_group(
    group: list[int],
    *,
    storage,
    device,
    threshold: float,
    void_color,
    allow_stuff_instances,
    allow_unknown_category,
    num_categories,
    object_ids,
    background_ids,
    key_segmentation_true: str,
    key_segmentation_pred: str,
    key_depth_true: str,
    key_depth_pred: str,
):
    r"""
    Computes DVPQ for a group of frames. This function is safe for use in a parallel context.
    """
    true_seg = storage.get_at(key_segmentation_true, group).to(
        device, non_blocking=True
    )
    pred_seg = storage.get_at(key_segmentation_pred, group).to(
        device, non_blocking=True
    )
    pred_seg = unipercept.vision.point.sparse_fill(pred_seg, pred_seg >= 0)
    true_dep = storage.get_at(key_depth_true, group).to(device, non_blocking=True)
    pred_dep = storage.get_at(key_depth_pred, group).to(device, non_blocking=True)
    assert pred_dep.shape == true_dep.shape, (pred_dep.shape, true_dep.shape)
    assert pred_seg.shape == true_seg.shape, (pred_seg.shape, true_seg.shape)
    assert true_dep.dtype == torch.float32, true_dep.dtype
    assert pred_dep.dtype == torch.float32, pred_dep.dtype

    invalid_depth_id = max(*object_ids, *background_ids) + 1

    if threshold > 0:
        # Apply same conversion as the reference implementation (ViP-DeepLab)
        true_dep = (true_dep * 256).int().float()
        pred_dep = (pred_dep * 256).int().float()
        valid_dep = true_dep > 0
        # valid_dep = torch.where(true_seg >= 0, valid_dep, False)
        abs_rel = torch.abs(pred_dep - true_dep) / true_dep
        abs_rel[~valid_dep] = 0.0
        pred_seg = torch.where(
            abs_rel > threshold,
            invalid_depth_id * tensors.PanopticMap.DIVISOR,
            pred_seg,
        )
        abs_rel_mean = abs_rel[valid_dep].mean()
    else:
        abs_rel_mean = torch.tensor(0, dtype=torch.float32, device=device)

    # Stack the group into one large image
    true_seg = rearrange(true_seg, "b h w -> (b h) w")
    pred_seg = rearrange(pred_seg, "b h w -> (b h) w")

    # Compute PQ
    background_ids = frozenset(
        list(background_ids) + [invalid_depth_id]
    )  # if not allow_unknown_category else background_ids,

    pred_seg = segmentation.preprocess_panoptic_quality(
        object_ids,
        background_ids,
        pred_seg,
        void_color=void_color,
        allow_unknown_category=allow_unknown_category,
    )
    true_seg = segmentation.preprocess_panoptic_quality(
        object_ids,
        background_ids,
        true_seg,
        void_color=void_color,
        allow_unknown_category=True,
    )

    return (
        *segmentation.compute_panoptic_quality_partial(
            pred_seg,
            true_seg,
            void_color=void_color,
            background_ids=(background_ids if not allow_stuff_instances else None),
            num_categories=num_categories + 1,
        ),
        abs_rel_mean,
    )


def _format_summary_number(num: T.Any, name: str) -> str:
    res = str(num)
    if "." in res:
        return res.replace(".", name)
    return res + name


def compute_stq(
    element,
    num_classes=19,
    max_ins=10000,
    ign_id=255,
    num_things=8,
    label_divisor=1e4,
    ins_divisor=1e7,
):
    import numpy as np

    y_pred, y_true = element
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # semantic eval
    semantic_label = y_true // max_ins
    semantic_prediction = y_pred // max_ins
    semantic_label = np.where(semantic_label != ign_id, semantic_label, num_classes)
    semantic_prediction = np.where(
        semantic_prediction != ign_id, semantic_prediction, num_classes
    )
    semantic_ids = np.reshape(semantic_label, [-1]) * label_divisor + np.reshape(
        semantic_prediction, [-1]
    )

    # instance eval
    instance_label = y_true % max_ins
    label_mask = np.less(semantic_label, num_things)
    prediction_mask = np.less(semantic_label, num_things)
    is_crowd = np.logical_and(instance_label == 0, label_mask)

    label_mask = np.logical_and(label_mask, np.logical_not(is_crowd))
    prediction_mask = np.logical_and(prediction_mask, np.logical_not(is_crowd))

    seq_preds = y_pred[prediction_mask]
    seg_labels = y_true[label_mask]

    non_crowd_intersection = np.logical_and(label_mask, prediction_mask)
    intersection_ids = (
        y_true[non_crowd_intersection] * ins_divisor + y_pred[non_crowd_intersection]
    )
    return semantic_ids, seq_preds, seg_labels, intersection_ids
