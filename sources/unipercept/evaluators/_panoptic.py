"""
Implements an evaluator for panoptic segmentation tasks.

Based on the Torchmetrics implementation of the Panoptic Quality metric.
"""
from __future__ import annotations

import dataclasses as D
import multiprocessing
import typing as T
import einops

import torch
import torch.types
import pandas as pd
from PIL import Image as pil_image
from tensordict import TensorDictBase
from tqdm import tqdm
from typing_extensions import override


from ..data.tensors import PanopticMap
from ..log import get_logger
from ._base import Evaluator, PlotMode
from . import helpers as H

if T.TYPE_CHECKING:
    from ..data.sets import Metadata
    from ..model import ModelOutput, InputData

__all__ = ["PanopticEvaluator", "TRUE_PANOPTIC", "PRED_PANOPTIC", "PanopticWriter"]

_logger = get_logger(__name__)
_ColorType: T.TypeAlias = tuple[
    int, int
]  # A (category_id, instance_id) tuple that uniquely identifies a panoptic segment.

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

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unipercept import get_info

        return cls(info=get_info(name), **kwargs)

    @override
    def update(self, storage: TensorDictBase, inputs: InputData, outputs: ModelOutput):
        super().update(storage, inputs, outputs)
        storage.setdefault(TRUE_PANOPTIC, outputs.truths.get("segmentations"), inplace=True)
        storage.setdefault(PRED_PANOPTIC, outputs.predictions.get("segmentations"), inplace=True)

    @override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        result = super().plot(storage)

        from unipercept.render.utils import draw_image_segmentation

        plot_keys = []
        for key, mode_attr in ((TRUE_PANOPTIC, "plot_true"), (PRED_PANOPTIC, "plot_pred")):
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            elif mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        for i in range(self.plot_samples):
            for key in plot_keys:
                result[f"{key}_{i}"] = draw_image_segmentation(storage.get_at(key, i).clone(), self.info)
        return result


@D.dataclass(kw_only=True)
class PanopticEvaluator(PanopticWriter):
    """
    Computes PQ metrics for panoptic segmentation tasks.
    """

    show_progress: bool = False
    show_summary: bool = True
    show_details: bool = False

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
        metrics["original"] = self.compute_pq(storage, **kwargs, allow_stuff_instances=True)
        metrics["balanced"] = self.compute_pq(storage, **kwargs, allow_stuff_instances=False)

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
        # device = torch.device("cpu")  # using multiprocessing

        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)

        # Loop over each sample independently: segments must not be matched across frames.
        sample_amt = storage.batch_size[0]
        # worker_amt = min(multiprocessing.cpu_count(), 16)
        assert sample_amt > 0, f"Batch size must be greater than zero, got {sample_amt=}"

        n_iter = range(sample_amt)
        if self.show_progress:
            n_iter = tqdm(n_iter, desc="accumulating pqs", dynamic_ncols=True, total=sample_amt)

        for n in n_iter:
            pred = storage.get_at(PRED_PANOPTIC, n).clone().to(device=device)
            true = storage.get_at(TRUE_PANOPTIC, n).clone().to(device=device)

            pred = _preprocess_mask(
                self.object_ids,
                self.background_ids,
                pred,
                void_color=void_color,
                allow_unknown_category=allow_unknown_category,
            )
            true = _preprocess_mask(
                self.object_ids,
                self.background_ids,
                true,
                void_color=void_color,
                allow_unknown_category=True,
            )
            result = _panoptic_quality_update_sample(
                pred,
                true,
                void_color=void_color,
                background_ids=self.background_ids if not allow_stuff_instances else None,
                num_categories=num_categories,
            )

            iou += result[0]
            tp += result[1]
            fp += result[2]
            fn += result[3]

        _logger.debug("Accumulating PQ-related metrics")

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
        th_mask: torch.Tensor = _isin(torch.arange(num_categories, device=device), list(self.object_ids))
        st_mask: torch.Tensor = _isin(torch.arange(num_categories, device=device), list(self.background_ids))

        for name, mask in [("all", tn_mask), ("thing", tn_mask & th_mask), ("stuff", tn_mask & st_mask)]:
            n_masked = n_valid[mask].sum().item()
            summary[name] = {
                "μPQ": pq[mask].mean().item(),
                "μSQ": rq[mask].mean().item(),
                "μRQ": fp[mask].mean().item(),
                "μIoU": iou[mask].mean().item(),
                "ΣTP": tp[mask].sum().item() / n_masked,
                "ΣFP": fp[mask].sum().item() / n_masked,
                "ΣFN": fn[mask].sum().item() / n_masked,
            }
        summary_df = self._tabulate(summary)
        if self.show_summary:
            self._show_table(
                f"Panoptic evaluation summary ({allow_stuff_instances=}, {allow_unknown_category=})", summary_df
            )

        # Detailed -- per class
        details = {}

        for i in range(pq.shape[0]):
            for semcls in self.info.semantic_classes.values():
                if semcls.unified_id == i:
                    name = f"{semcls.name}".lower().replace(" ", "_")
                    break
            else:
                name = f"unknown({i})"

            n_masked = n_valid[i].sum().item()
            details[name] = {
                "μPQ": pq[i].mean().item(),
                "μSQ": rq[i].mean().item(),
                "μRQ": fp[i].mean().item(),
                "μIoU": iou[i].mean().item(),
                "ΣTP": tp[i].sum().item() / n_masked,
                "ΣFP": fp[i].sum().item() / n_masked,
                "ΣFN": fn[i].sum().item() / n_masked,
            }
        details_df = self._tabulate(details)
        if self.show_details:
            self._show_table(
                f"Panoptic evaluation details({allow_stuff_instances=}, {allow_unknown_category=})", details_df
            )

        return summary | details

    def _show_table(self, msg: str, tab: pd.DataFrame) -> None:
        tab_fmt = tab.to_markdown(index=False)
        _logger.info(f"%s:\n%s", msg, tab_fmt)

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


def _nested_tuple(nested_list: list) -> tuple:
    """Construct a nested tuple from a nested list.

    Args:
        nested_list: The nested list to convert to a nested tuple.

    Returns:
        A nested tuple with the same content.

    """
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _to_tuple(t: torch.Tensor) -> tuple:
    """Convert a tensor into a nested tuple.

    Args:
        t: The tensor to convert.

    Returns:
        A nested tuple with the same content.

    """
    return _nested_tuple(t.tolist())


def _get_color_areas(inputs: torch.Tensor) -> dict[tuple, torch.Tensor]:
    """Measure the size of each instance.

    Args:
        inputs: the input tensor containing the colored pixels.

    Returns:
        A dictionary specifying the `(category_id, instance_id)` and the corresponding number of occurrences.

    """
    unique_keys, unique_keys_area = torch.unique(inputs, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_to_tuple(unique_keys), unique_keys_area))


def _get_void_color(things: T.FrozenSet[int], stuffs: T.FrozenSet[int]) -> tuple[int, int]:
    """Get an unused color ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new color ID that does not belong to things nor stuffs.

    """
    unused_category_id = 1 + max([0, *list(things), *list(stuffs)])
    return unused_category_id, 0


def _get_category_id_to_continuous_id(things: T.FrozenSet[int], stuffs: T.FrozenSet[int]) -> dict[int, int]:
    """Convert original IDs to continuous IDs.

    Args:
        things: All unique IDs for things classes.
        stuffs: All unique IDs for stuff classes.

    Returns:
        A mapping from the original category IDs to continuous IDs (i.e., 0, 1, 2, ...).

    """
    # things metrics are stored with a continuous id in [0, len(things)),
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continuous id in [len(things), len(things) + len(stuffs))
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuffs)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id


def _isin(arr: torch.Tensor, values: list) -> torch.Tensor:
    """Check if all values of an arr are in another array. Implementation of torch.isin to support pre 0.10 version.

    Args:
        arr: the torch tensor to check for availabilities
        values: the values to search the tensor for.

    Returns:
        a bool tensor of the same shape as :param:`arr` indicating for each
        position whether the element of the tensor is in :param:`values`

    """
    return (arr[..., None] == arr.new(values)).any(-1)


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

    mask_stuffs = _isin(out[:, 0], list(stuffs))
    mask_things = _isin(out[:, 0], list(things))

    if not allow_unknown_category and not torch.all(mask_things | mask_stuffs):
        raise ValueError(f"Unknown categories found: {out[~(mask_things|mask_stuffs)]}")

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
    target_areas = T.cast(dict[_ColorType, torch.Tensor], _get_color_areas(flatten_target))
    # intersection matrix of shape [num_pixels, 2, 2]
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    assert intersection_matrix.shape == (flatten_preds.shape[0], 2, 2)
    intersection_areas = T.cast(
        dict[tuple[_ColorType, _ColorType], torch.Tensor], _get_color_areas(intersection_matrix)
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
        iou = _calculate_iou(pred_color, target_color, pred_areas, target_areas, intersection_areas, void_color)
        sem_id = target_color[0]
        if target_color[0] not in background_ids and iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[sem_id] += iou
            true_positives[sem_id] += 1
        elif target_color[0] in background_ids and iou > 0:
            iou_sum[sem_id] += iou

    for cat_id in _filter_false_negatives(target_areas, target_segment_matched, intersection_areas, void_color):
        if cat_id not in background_ids:
            false_negatives[cat_id] += 1

    for cat_id in _filter_false_positives(pred_areas, pred_segment_matched, intersection_areas, void_color):
        if cat_id not in background_ids:
            false_positives[cat_id] += 1

    for cat_id, _ in target_areas:
        if cat_id in background_ids:
            true_positives[cat_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives


def _compute_stq(element, num_classes=19, max_ins=10000, ign_id=255, num_things=8, label_divisor=1e4, ins_divisor=1e7):
    import numpy as np

    y_pred, y_true = element
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # semantic eval
    semantic_label = y_true // max_ins
    semantic_prediction = y_pred // max_ins
    semantic_label = np.where(semantic_label != ign_id, semantic_label, num_classes)
    semantic_prediction = np.where(semantic_prediction != ign_id, semantic_prediction, num_classes)
    semantic_ids = np.reshape(semantic_label, [-1]) * label_divisor + np.reshape(semantic_prediction, [-1])

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
    intersection_ids = y_true[non_crowd_intersection] * ins_divisor + y_pred[non_crowd_intersection]
    return semantic_ids, seq_preds, seg_labels, intersection_ids
