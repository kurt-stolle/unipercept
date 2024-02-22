"""
Implements an evaluator for panoptic segmentation tasks.

Based on the Torchmetrics implementation of the Panoptic Quality metric.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses as D
import enum as E
import functools
import typing as T

import pandas as pd
import torch
import torch.multiprocessing as M
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase
from tqdm import tqdm
from typing_extensions import override

from unipercept.data.tensors import PanopticMap
from unipercept.evaluators import helpers as H
from unipercept.evaluators._base import Evaluator, PlotMode
from unipercept.log import get_logger
from unipercept.state import check_main_process, cpus_available

if T.TYPE_CHECKING:
    from ..data.sets import Metadata

__all__ = ["PanopticEvaluator", "TRUE_PANOPTIC", "PRED_PANOPTIC", "PanopticWriter"]

_logger = get_logger(__name__)
_ColorType: T.TypeAlias = tuple[
    int, int
]  # A (category_id, instance_id) tuple that uniquely identifies a panoptic segment.

VALID_PANOPTIC: T.Final[str] = "valid_panoptic"
TRUE_PANOPTIC: T.Final[str] = "true_panoptic"
PRED_PANOPTIC: T.Final[str] = "pred_panoptic"


@D.dataclass(kw_only=True)
class PanopticWriter(Evaluator):
    """
    Stores and optionally renders panoptic segmentation outputs.
    """

    info: Metadata = D.field(repr=False)

    plot_samples: int = 1
    plot_true: PlotMode = PlotMode.ONCE
    plot_pred: PlotMode = PlotMode.ALWAYS

    true_key = ("captures", "segmentations")
    true_group_index = -1  # the most recent group, assuming temporal ordering
    pred_key = "segmentations"

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unipercept import get_info

        return cls(info=get_info(name), **kwargs)

    @override
    def update(
        self, storage: TensorDictBase, inputs: TensorDictBase, outputs: TensorDictBase
    ):
        """
        Stores the panoptic segmentation predictions and ground truths in storage for later evaluation.
        """
        super().update(storage, inputs, outputs)

        storage_keys = storage.keys(include_nested=True, leaves_only=True)
        assert storage_keys is not None, "Storage keys are empty"
        if (
            TRUE_PANOPTIC in storage_keys
            and PRED_PANOPTIC in storage_keys
            and VALID_PANOPTIC in storage_keys
        ):
            return

        pred = outputs.get(self.pred_key)
        if pred is None:
            raise RuntimeError(f"Panoptic segmentation output not found in {outputs=}")

        true: torch.Tensor = inputs.get(self.true_key, default=None)
        if true is None:  # Generate dummy values for robust evaluation downstream
            true = torch.full_like(pred, PanopticMap.IGNORE, dtype=torch.long)
        else:
            true = true[:, self.true_group_index, ...]

        valid = (true != PanopticMap.IGNORE).any(dim=(-1)).any(dim=-1)

        for key, item in (
            (TRUE_PANOPTIC, true),
            (PRED_PANOPTIC, pred),
            (VALID_PANOPTIC, valid),
        ):
            storage.set(key, item, inplace=True)

    @override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        result = super().plot(storage)

        from unipercept.render import draw_image_segmentation

        plot_keys = []
        for key, mode_attr in (
            (TRUE_PANOPTIC, "plot_true"),
            (PRED_PANOPTIC, "plot_pred"),
        ):
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            elif mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        for i in range(self.plot_samples):
            for key in plot_keys:
                result[f"{key}_{i}"] = draw_image_segmentation(
                    storage.get_at(key, i), self.info
                )
        return result


class PQMetrics(T.NamedTuple):
    pq: torch.Tensor
    sq: torch.Tensor
    rq: torch.Tensor
    iou: torch.Tensor
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor


class PQDefinition(E.IntEnum):
    ORIGINAL = E.auto()
    BALANCED = E.auto()


@D.dataclass(kw_only=True)
class PanopticEvaluator(PanopticWriter):
    """
    Computes PQ metrics for panoptic segmentation tasks.
    """

    show_progress: bool = True
    show_summary: bool = True
    show_details: bool = False
    report_details: bool = False

    pq_definition: PQDefinition = PQDefinition.ORIGINAL

    @classmethod
    @override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @property
    def object_ids(self) -> frozenset[int]:
        return self.info.object_ids

    @property
    def background_ids(self) -> frozenset[int]:
        return self.info.background_ids

    @override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        metrics = super().compute(storage, **kwargs)

        if self.pq_definition & PQDefinition.ORIGINAL:
            metrics["original"] = self.compute_pq(
                storage, **kwargs, allow_stuff_instances=True
            )
        if self.pq_definition & PQDefinition.BALANCED:
            metrics["balanced"] = self.compute_pq(
                storage, **kwargs, allow_stuff_instances=False
            )

        if len(metrics) == 0:
            raise ValueError("No PQ definition selected.")

        return metrics

    def compute_pq(
        self,
        storage: TensorDictBase,
        *,
        device: torch.types.Device,
        allow_stuff_instances: bool = False,
        allow_unknown_category: bool = False,
    ) -> dict[str, T.Any]:
        """
        Calculate stat scores required to compute the metric for a full batch.
        """
        void_color = _get_void_color(self.object_ids, self.background_ids)
        device = torch.device("cpu")  # using multiprocessing
        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)
        sample_amt = storage.batch_size[0]
        assert (
            sample_amt > 0
        ), f"Batch size must be greater than zero, got {sample_amt=}"
        compute_at = functools.partial(
            _compute_at,
            storage=storage,
            object_ids=self.object_ids,
            background_ids=self.background_ids,
            device=device,
            allow_unknown_category=allow_unknown_category,
            void_color=void_color,
            allow_stuff_instances=allow_stuff_instances,
            num_categories=num_categories,
        )
        progress_bar = tqdm(
            desc="Computing panoptic metrics",
            dynamic_ncols=True,
            total=sample_amt,
            disable=not check_main_process(local=True) or not self.show_progress,
        )
        #mp_context = M.get_context("spawn" if device.type != "cpu" else None)
        #with concurrent.futures.ProcessPoolExecutor(
        #    min(cpus_available(), M.cpu_count() // 2, 32), mp_context=mp_context
        #) as pool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for result in pool.map(compute_at, range(sample_amt)):
                progress_bar.update(1)
                if result is None:
                    continue
                iou += result[0]
                tp += result[1]
                fp += result[2]
                fn += result[3]
        progress_bar.close()
        # Compute PQ = SQ * RQ
        sq = H.stable_divide(iou, tp)
        rq = H.stable_divide(tp, tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq

        # Convert to percentages
        metrics = PQMetrics(pq * 100, sq * 100, rq * 100, iou, tp, fp, fn)
        n_valid = tp + fp + fn

        summary = self._create_summary_report(
            metrics,
            n_valid=n_valid,
            allow_unknown_category=allow_unknown_category,
            allow_stuff_instances=allow_stuff_instances,
            num_categories=num_categories,
        )
        details = self._create_detail_report(
            metrics,
            n_valid=n_valid,
            allow_unknown_category=allow_unknown_category,
            allow_stuff_instances=allow_stuff_instances,
        )

        if self.report_details:
            out = summary | details
        else:
            out = summary

        return out

    def _create_summary_report(
        self,
        metrics: PQMetrics,
        *,
        n_valid,
        allow_unknown_category,
        allow_stuff_instances,
        num_categories,
    ):
        tn_mask: torch.Tensor = n_valid > 0
        th_mask: torch.Tensor = H.isin(
            torch.arange(num_categories, device=metrics.pq.device),
            list(self.object_ids),
        )
        st_mask: torch.Tensor = H.isin(
            torch.arange(num_categories, device=metrics.pq.device),
            list(self.background_ids),
        )
        summary = {}
        for name, mask in [
            ("all", tn_mask),
            ("thing", tn_mask & th_mask),
            ("stuff", tn_mask & st_mask),
        ]:
            n_masked = n_valid[mask].sum().item()
            if n_masked == 0:
                summary[name] = {
                    "PQ": 1.0,
                    "SQ": 1.0,
                    "RQ": 1.0,
                    "IoU": 0.0,
                    "TP": 0.0,
                    "FP": 0.0,
                    "FN": 0.0,
                }
            else:
                summary[name] = {
                    "PQ": metrics.pq[mask].mean().item(),
                    "SQ": metrics.sq[mask].mean().item(),
                    "RQ": metrics.rq[mask].mean().item(),
                    "IoU": metrics.iou[mask].mean().item(),
                    "TP": metrics.tp[mask].sum().item() / n_masked,
                    "FP": metrics.fp[mask].sum().item() / n_masked,
                    "FN": metrics.fn[mask].sum().item() / n_masked,
                }
        if self.show_summary:
            df = _tabulate(summary)
            msg = f"Panoptic summary ({allow_stuff_instances=}, {allow_unknown_category=})"
            self._show_table(msg, df)
        return summary

    def _create_detail_report(
        self,
        metrics,
        *,
        n_valid,
        allow_unknown_category,
        allow_stuff_instances,
    ):
        details = {}
        for i in range(metrics.pq.shape[0]):
            for semcls in self.info.semantic_classes.values():
                if semcls.unified_id == i:
                    name = f"{semcls.name}".lower().replace(" ", "_")
                    break
            else:
                name = f"unknown({i})"

            n_masked = n_valid[i].sum().item()
            if n_masked == 0:
                details[name] = {
                    "PQ": 1.0,
                    "SQ": 1.0,
                    "RQ": 1.0,
                    "IoU": 0.0,
                    "TP": 0.0,
                    "FP": 0.0,
                    "FN": 0.0,
                }
            else:
                details[name] = {
                    "PQ": metrics.pq[i].mean().item(),
                    "SQ": metrics.sq[i].mean().item(),
                    "RQ": metrics.rq[i].mean().item(),
                    "IoU": metrics.iou[i].mean().item(),
                    "TP": metrics.tp[i].sum().item() / n_masked,
                    "FP": metrics.fp[i].sum().item() / n_masked,
                    "FN": metrics.fn[i].sum().item() / n_masked,
                }
        if self.show_details:
            df = _tabulate(details)
            msg = f"Panoptic details ({allow_stuff_instances=}, {allow_unknown_category=})"
            self._show_table(msg, df)
        return details


def _tabulate(result: dict[str, dict[str, float]]) -> pd.DataFrame:
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


def _compute_at(
    n,
    *,
    storage,
    object_ids,
    background_ids,
    device,
    allow_unknown_category,
    void_color,
    allow_stuff_instances,
    num_categories,
):
    valid = storage.get_at(VALID_PANOPTIC, n).item()
    if not valid:
        return None
    pred = storage.get_at(PRED_PANOPTIC, n).to(device=device, non_blocking=True)
    true = storage.get_at(TRUE_PANOPTIC, n).to(device=device, non_blocking=True)
    pred = _preprocess_mask(
        object_ids,
        background_ids,
        pred,
        void_color=void_color,
        allow_unknown_category=allow_unknown_category,
    )
    true = _preprocess_mask(
        object_ids,
        background_ids,
        true,
        void_color=void_color,
        allow_unknown_category=True,
    )
    result = _panoptic_quality_update_sample(
        pred,
        true,
        void_color=void_color,
        background_ids=background_ids if not allow_stuff_instances else None,
        num_categories=num_categories,
    )
    return result


def _nested_tuple(nested_list: list) -> tuple:
    r"""
    Construct a nested tuple from a nested list.

    Parameters
    ----------
    nested_list:
        The nested list to convert to a nested tuple.

    Returns
    -------
        A nested tuple with the same content.

    """
    return (
        tuple(map(_nested_tuple, nested_list))
        if isinstance(nested_list, list)
        else nested_list
    )


def _to_tuple(t: torch.Tensor) -> tuple:
    r"""
    Convert a tensor into a nested tuple.

    Parameters
    ----------
    t:
        The tensor to convert.

    Returns
    -------
        A nested tuple with the same content.
    """
    return _nested_tuple(t.tolist())


def _get_color_areas(inputs: torch.Tensor) -> dict[tuple, torch.Tensor]:
    r"""
    Measure the size of each instance.

    Parameters
    ----------
    inputs:
        The input tensor containing the colored pixels.

    Returns
    -------
        A dictionary specifying the `(category_id, instance_id)` and the corresponding
        number of occurrences.

    """
    unique_keys, unique_keys_area = torch.unique(inputs, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_to_tuple(unique_keys), unique_keys_area))


def _get_void_color(
    things: T.FrozenSet[int], stuffs: T.FrozenSet[int]
) -> tuple[int, int]:
    r"""
    Get an unused color ID.

    Parameters
    ----------
    things:
        The set of category IDs for things.
    stuffs:
        The set of category IDs for stuffs.

    Returns
    -------
        A new color ID that does not belong to things nor stuffs.
    """
    unused_category_id = 1 + max([0, *list(things), *list(stuffs)])
    return unused_category_id, 0


def _get_category_id_to_continuous_id(
    things: T.FrozenSet[int], stuffs: T.FrozenSet[int]
) -> dict[int, int]:
    r"""
    Convert original IDs to continuous IDs.

    Parameters
    ----------
    things:
        All unique IDs for things classes.
    stuffs:
        All unique IDs for stuff classes.

    Returns
    -------
        A mapping from the original category IDs to continuous IDs (i.e., 0, 1, 2, ...).

    """
    # things metrics are stored with a continuous id in [0, len(things)),
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continuous id in [len(things), len(things) + len(stuffs))
    stuff_id_to_continuous_id = {
        stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuffs)
    }
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id


def _preprocess_mask(
    things: T.FrozenSet[int],
    stuffs: T.FrozenSet[int],
    inputs: PanopticMap,
    void_color: tuple[int, int],
    allow_unknown_category: bool,
) -> torch.Tensor:
    """
    Preprocesses an input tensor for metric calculation. Inputs should be **unbatched**.

    Parameters
    ----------
    things
        All category IDs for things classes.
    stuffs
        All category IDs for stuff classes.
    inputs
        The input tensor.
    void_color
        An additional color that is masked out during metrics calculation.
    allow_unknown_category
        If true, unknown category IDs are mapped to "void". Otherwise, an exception is raised if they occur.

    Returns
    -------
    The preprocessed input tensor flattened along the spatial dimensions.
    """

    # Check that the union of things and stuff is disjoint
    assert len(things & stuffs) == 0, "Things and stuffs must be disjoint"
    inputs = inputs.detach().as_subclass(PanopticMap)

    # Remove instance IDs of stuff classes
    inputs.remove_instances_(stuffs)

    # Flatten the spatial dimensions of the input tensor, e.g., (B, H, W, C) -> (B*H*W, C).
    out = inputs.to_parts()
    out = torch.flatten(out, 0, -2)
    assert out.ndim == 2, out.shape

    mask_stuffs = H.isin(out[:, 0], list(stuffs))
    mask_things = H.isin(out[:, 0], list(things))

    if not allow_unknown_category and not torch.all(mask_things | mask_stuffs):
        raise ValueError(
            f"Unknown categories found: {out[~(mask_things|mask_stuffs)].unique().cpu().tolist()}"
        )

    # Set unknown categories to void color
    out[~(mask_things | mask_stuffs)] = out.new(void_color)

    return out


def _calculate_iou(
    pred_color: _ColorType,
    target_color: _ColorType,
    pred_areas: dict[_ColorType, torch.Tensor],
    target_areas: dict[_ColorType, torch.Tensor],
    intersection_areas: dict[tuple[_ColorType, _ColorType], torch.Tensor],
    void_color: _ColorType,
) -> torch.Tensor:
    """
    Helper function that calculates the IoU from precomputed areas of segments and their intersections.
    """
    if pred_color[0] != target_color[0]:
        raise ValueError(
            "Attempting to compute IoU on segments with different category ID: "
            f"pred {pred_color[0]}, target {target_color[0]}"
        )
    if pred_color == void_color:
        raise ValueError("Attempting to compute IoU on a void segment.")
    intersection = intersection_areas[(pred_color, target_color)]
    pred_area = pred_areas[pred_color]
    target_area = target_areas[target_color]
    pred_void_area = intersection_areas.get((pred_color, void_color), 0)
    void_target_area = intersection_areas.get((void_color, target_color), 0)
    union = pred_area - pred_void_area + target_area - void_target_area - intersection
    return intersection / union


def _filter_false_negatives(
    target_areas: dict[_ColorType, torch.Tensor],
    target_segment_matched: T.Set[_ColorType],
    intersection_areas: dict[tuple[_ColorType, _ColorType], torch.Tensor],
    void_color: tuple[int, int],
) -> T.Iterator[int]:
    """
    Filter false negative segments and yield their category IDs.

    False negatives occur when a ground truth segment is not matched with a prediction.
    Areas that are mostly void in the prediction are ignored.
    """
    false_negative_colors = set(target_areas) - target_segment_matched
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] <= 0.5:
            yield target_color[0]


def _filter_false_positives(
    pred_areas: dict[_ColorType, torch.Tensor],
    pred_segment_matched: T.Set[_ColorType],
    intersection_areas: dict[tuple[_ColorType, _ColorType], torch.Tensor],
    void_color: tuple[int, int],
) -> T.Iterator[int]:
    """
    Filter false positive segments and yield their category IDs.

    False positives occur when a predicted segment is not matched with a corresponding target one.
    Areas that are mostly void in the target are ignored.
    """
    false_positive_colors = set(pred_areas) - pred_segment_matched
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] <= 0.5:
            yield pred_color[0]


def _panoptic_quality_update_sample(
    flatten_preds: torch.Tensor,
    flatten_target: torch.Tensor,
    void_color: tuple[int, int],
    num_categories: int,
    background_ids: T.Optional[T.FrozenSet[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate stat scores required to compute the metric for a single sample.
    """
    background_ids = background_ids or frozenset()
    device = flatten_preds.device

    iou_sum = torch.zeros(num_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(num_categories, dtype=torch.int, device=device)

    # calculate the area of each prediction, ground truth and pairwise intersection.
    # NOTE: mypy needs `cast()` because the annotation for `_get_color_areas` is too generic.
    pred_areas = T.cast(dict[_ColorType, torch.Tensor], _get_color_areas(flatten_preds))
    target_areas = T.cast(
        dict[_ColorType, torch.Tensor], _get_color_areas(flatten_target)
    )
    # intersection matrix of shape [num_pixels, 2, 2]
    intersection_matrix = torch.transpose(
        torch.stack((flatten_preds, flatten_target), -1), -1, -2
    )
    assert intersection_matrix.shape == (flatten_preds.shape[0], 2, 2)
    intersection_areas = T.cast(
        dict[tuple[_ColorType, _ColorType], torch.Tensor],
        _get_color_areas(intersection_matrix),
    )

    # select intersection of things of same category with iou > 0.5
    pred_segment_matched = set()
    target_segment_matched = set()
    for pred_color, target_color in intersection_areas.keys():
        # test only non void, matching category
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        iou = _calculate_iou(
            pred_color,
            target_color,
            pred_areas,
            target_areas,
            intersection_areas,
            void_color,
        )
        sem_id = target_color[0]
        if target_color[0] not in background_ids and iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[sem_id] += iou
            true_positives[sem_id] += 1
        elif target_color[0] in background_ids and iou > 0:
            iou_sum[sem_id] += iou

    for cat_id in _filter_false_negatives(
        target_areas, target_segment_matched, intersection_areas, void_color
    ):
        if cat_id not in background_ids:
            false_negatives[cat_id] += 1

    for cat_id in _filter_false_positives(
        pred_areas, pred_segment_matched, intersection_areas, void_color
    ):
        if cat_id not in background_ids:
            false_positives[cat_id] += 1

    for cat_id, _ in target_areas:
        if cat_id in background_ids:
            true_positives[cat_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives


def _compute_stq(
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
