"""
Implements an evaluator for panoptic segmentation tasks.
"""
from __future__ import annotations

import dataclasses as D
import functools
import multiprocessing
import typing as T
import warnings

import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase
from tqdm import tqdm
from typing_extensions import override

from ..data.tensors import PanopticMap
from ..utils.logutils import get_logger
from ._base import Evaluator

if T.TYPE_CHECKING:
    from ..data.sets import Metadata
    from ..model import ModelOutput

__all__ = ["PanopticEvaluator"]

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
    render_samples: int = 1

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unicore import catalog

        return cls(info=catalog.get_info(name), **kwargs)

    @override
    def update(self, storage: TensorDictBase, outputs: ModelOutput):
        storage.setdefault(TRUE_PANOPTIC, outputs.truths.get("segmentations"), inplace=True)
        storage.setdefault(PRED_PANOPTIC, outputs.predictions.get("segmentations"), inplace=True)

    @override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        return {}

    @override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        from unipercept.render.utils import draw_image_segmentation

        result = {}
        for i in range(self.render_samples):
            for key in (PRED_PANOPTIC, TRUE_PANOPTIC):
                result[f"{key}_{i}"] = draw_image_segmentation(storage.get_at(key, i).clone(), self.info)
        return result


@D.dataclass(kw_only=True)
class PanopticEvaluator(PanopticWriter):
    """
    Computes PQ metrics for panoptic segmentation tasks.
    """

    show_progress: bool = False

    @classmethod
    @override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @property
    def thing_cats(self) -> set[int]:
        return set(self.info.thing_ids)

    @property
    def stuff_cats(self) -> set[int]:
        return set(self.info.stuff_ids) - self.thing_cats

    @override
    def compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        metrics = super().compute(storage, **kwargs)
        metrics["pq_ori"] = self.compute_pq(storage, **kwargs, use_modified=False)
        metrics["pq_mod"] = self.compute_pq(storage, **kwargs, use_modified=True)

        return metrics

    def compute_pq(
        self,
        storage: TensorDictBase,
        *,
        device: torch.types.Device,
        use_modified: bool = False,
        allow_unknown_category: bool = False,
    ) -> dict[str, T.Any]:
        """
        Calculate stat scores required to compute the metric for a full batch.

        Computed scores: iou sum, true positives, false positives, false negatives.

        Args:
            flatten_preds: A flattened prediction tensor, shape (B, num_points, 2).
            flatten_target: A flattened target tensor, shape (B, num_points, 2).
            cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs.
            void_color: an additional, unused color.
            modified_metric_stuffs: T.Set of stuff category IDs for which the PQ metric is computed using the "modified"
                formula. If not specified, the original formula is used for all categories.

        Returns:
            - IOU Sum
            - True positives
            - False positives
            - False negatives

        """
        void_color = _get_void_color(self.thing_cats, self.stuff_cats)
        cat_id_to_continuous_id = _get_category_id_to_continuous_id(self.thing_cats, self.stuff_cats)

        # device = torch.device("cpu")  # using multiprocessing

        num_categories = len(cat_id_to_continuous_id)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)

        # Loop over each sample independently: segments must not be matched across frames.
        sample_amt = storage.batch_size[0]
        worker_amt = min(multiprocessing.cpu_count(), 16)

        assert sample_amt > 0, f"Batch size must be greater than zero, got {sample_amt=}"

        _logger.debug(f"Starting evaluation of {sample_amt} samples in {worker_amt} workers")

        n_iter = range(sample_amt)
        if self.show_progress:
            n_iter = tqdm(n_iter, desc="accumulating pqs", dynamic_ncols=True, total=sample_amt)

        for n in n_iter:
            pred = storage.get_at(PRED_PANOPTIC, n).clone().to(device=device)
            true = storage.get_at(TRUE_PANOPTIC, n).clone().to(device=device)

            pred = _preprocess_mask(
                self.thing_cats,
                self.stuff_cats,
                pred,
                void_color=void_color,
                allow_unknown_category=allow_unknown_category,
            )
            true = _preprocess_mask(
                self.thing_cats,
                self.stuff_cats,
                true,
                void_color=void_color,
                allow_unknown_category=True,
            )
            result = _panoptic_quality_update_sample(
                pred,
                true,
                cat_id_to_continuous_id,
                void_color=void_color,
                stuffs_modified_metric=self.stuff_cats if use_modified else None,
            )

            iou += result[0]
            tp += result[1]
            fp += result[2]
            fn += result[3]

        den = (tp + 0.5 * fp + 0.5 * fn).double()
        pq = torch.where(den > 0.0, iou / den, 0.0) * 100.0

        return {
            "pq": pq.double().mean().item(),
            "iou": iou.double().mean().item(),
            "tp": tp.double().mean().item(),
            "fp": fp.double().mean().item(),
            "fn": fn.double().mean().item(),
        }


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


def _get_void_color(things: T.Set[int], stuffs: T.Set[int]) -> tuple[int, int]:
    """Get an unused color ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new color ID that does not belong to things nor stuffs.

    """
    unused_category_id = 1 + max([0, *list(things), *list(stuffs)])
    return unused_category_id, 0


def _get_category_id_to_continuous_id(things: T.Set[int], stuffs: T.Set[int]) -> dict[int, int]:
    """Convert original IDs to continuous IDs.

    Args:
        things: All unique IDs for things classes.
        stuffs: All unique IDs for stuff classes.

    Returns:
        A mapping from the original category IDs to continuous IDs (i.e., 0, 1, 2, ...).

    """
    # things metrics are stored with a continuous id in [0, len(things)[,
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continuous id in [len(things), len(things) + len(stuffs)[
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
    things: T.Set[int],
    stuffs: T.Set[int],
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

    # flatten the spatial dimensions of the input tensor, e.g., (B, H, W, C) -> (B*H*W, C).
    out = inputs.detach().as_subclass(PanopticMap).to_parts()
    out = torch.flatten(out, 0, -2)
    mask_stuffs = _isin(out[:, 0], list(stuffs))
    mask_things = _isin(out[:, 0], list(things))
    # reset instance IDs of stuffs
    mask_stuffs_instance = torch.stack([torch.zeros_like(mask_stuffs), mask_stuffs], dim=-1)
    out[mask_stuffs_instance] = 0
    if not allow_unknown_category and not torch.all(mask_things | mask_stuffs):
        raise ValueError(f"Unknown categories found: {out[~(mask_things|mask_stuffs)]}")
    # set unknown categories to void color
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

    Args:
        pred_color: The `(category_id, instance_id)`, or "color", of a predicted segment that is being matched with a
            target segment.
        target_color: The `(category_id, instance_id)`, or "color", of a ground truth segment that is being matched
            with a predicted segment.
        pred_areas: Mapping from colors of the predicted segments to their extents.
        target_areas: Mapping from colors of the ground truth segments to their extents.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Returns:
        The calculated IoU as a torch.torch.Tensor containing a single scalar value.

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

    Args:
        target_areas: Mapping from colors of the ground truth segments to their extents.
        target_segment_matched: T.Set of ground truth segments that have been matched to a prediction.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false negatives.

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

    Args:
        pred_areas: Mapping from colors of the predicted segments to their extents.
        pred_segment_matched: T.Set of predicted segments that have been matched to a ground truth.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false positives.

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
    cat_id_to_continuous_id: dict[int, int],
    void_color: tuple[int, int],
    stuffs_modified_metric: T.Optional[T.Set[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate stat scores required to compute the metric **for a single sample**.

    Computed scores: iou sum, true positives, false positives, false negatives.

    NOTE: For the modified PQ case, this implementation uses the `true_positives` output tensor to aggregate the actual
        TPs for things classes, but the number of target segments for stuff classes.
        The `iou_sum` output tensor, instead, aggregates the IoU values at different thresholds (i.e., 0.5 for things
        and 0 for stuffs).
        This allows seamlessly using the same `.compute()` method for both PQ variants.

    Args:
        flatten_preds: A flattened prediction tensor referring to a single sample, shape (num_points, 2).
        flatten_target: A flattened target tensor referring to a single sample, shape (num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs
        void_color: an additional, unused color.
        stuffs_modified_metric: T.Set of stuff category IDs for which the PQ metric is computed using the "modified"
            formula. If not specified, the original formula is used for all categories.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives.

    """
    stuffs_modified_metric = stuffs_modified_metric or set()
    device = flatten_preds.device
    num_categories = len(cat_id_to_continuous_id)
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
        continuous_id = cat_id_to_continuous_id[target_color[0]]
        if target_color[0] not in stuffs_modified_metric and iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1
        elif target_color[0] in stuffs_modified_metric and iou > 0:
            iou_sum[continuous_id] += iou

    for cat_id in _filter_false_negatives(target_areas, target_segment_matched, intersection_areas, void_color):
        if cat_id not in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            false_negatives[continuous_id] += 1

    for cat_id in _filter_false_positives(pred_areas, pred_segment_matched, intersection_areas, void_color):
        if cat_id not in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            false_positives[continuous_id] += 1

    for cat_id, _ in target_areas:
        if cat_id in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            true_positives[continuous_id] += 1

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
