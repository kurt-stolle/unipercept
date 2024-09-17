"""
Implements an evaluator for segmentation tasks.

Computes metrics like PQ, SQ, RQ, and mIoU for segmentation tasks.
"""

from __future__ import annotations

import abc
import concurrent.futures
import dataclasses as D
import enum as E
import functools
import json
import typing as T

import pandas as pd
import torch
import torch.types
import typing_extensions as TX
from PIL import Image as pil_image
from tensordict import TensorDictBase
from torch import Tensor, nn

from unipercept import file_io, state
from unipercept.data.tensors import PanopticMap, PanopticMapLike
from unipercept.data.types.coco import COCOResultPanoptic

from ._base import Evaluator, EvaluatorComputeKWArgs, PlotMode, StoragePrefix
from ._common import isin, nonzero_divide

if T.TYPE_CHECKING:
    from unipercept.model import InputData


class SegmentationTask(E.StrEnum):
    PANOPTIC_SEGMENTATION = E.auto()
    SEMANTIC_SEGMENTATION = E.auto()
    INSTANCE_SEGMENTATION = E.auto()


@D.dataclass(kw_only=True)
class SegmentationWriter(Evaluator, metaclass=abc.ABCMeta):
    """
    Stores and optionally renders panoptic segmentation outputs.
    """

    segmentation_plot_samples: int = D.field(
        default=1, metadata={"help": "Number of samples to plot"}
    )
    segmentation_plot_true: PlotMode = PlotMode.NEVER
    segmentation_plot_pred: PlotMode = PlotMode.ALWAYS
    segmentation_requires_true: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the target segmentation is not found"},
    )
    segmentation_requires_pred: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the predicted segmentation is not found"},
    )
    segmentation_key: str = D.field(
        default=None,  # type: ignore
        metadata={
            "help": (
                "The key for the segmentation predictions in the output TensorDict. "
                "If None, the evaluator will use the default key for the configured task."
            )
        },
    )
    segmentation_task: SegmentationTask = D.field(
        default=SegmentationTask.PANOPTIC_SEGMENTATION,
        metadata={
            "help": (
                "The type of segmentation task. Used to determine the default keys and "
                "validate the predicated output. Ground truths are always in panoptic "
                "segmentation format."
            )
        },
    )

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        if isinstance(self.segmentation_task, str):
            self.segmentation_task = SegmentationTask(self.segmentation_task)
        if self.segmentation_key is None:
            self.segmentation_key = self.segmentation_task.value

    @property
    def segmentation_key_pred(self) -> str:
        assert self.segmentation_key is not None
        return self._get_storage_key(self.segmentation_key, StoragePrefix.PRED)

    @property
    def segmentation_key_true(self) -> str:
        assert self.segmentation_key is not None
        return self._get_storage_key(self.segmentation_key, StoragePrefix.TRUE)

    @property
    def segmentation_key_valid(self) -> str:
        assert self.segmentation_key is not None
        return self._get_storage_key(self.segmentation_key, StoragePrefix.VALID)

    @TX.override
    def _update(
        self,
        storage: TensorDictBase,
        inputs: InputData,
        outputs: TensorDictBase,
        **kwargs,
    ):
        """
        See :meth:`Evaluator.update`.
        """
        super()._update(storage, inputs, outputs, **kwargs)

        target_keys = {
            self.segmentation_key_valid,
            self.segmentation_key_pred,
            self.segmentation_key_valid,
        }
        storage_keys = storage.keys(include_nested=True, leaves_only=True)
        assert storage_keys is not None
        if target_keys.issubset(storage_keys):
            return

        input_images = inputs.captures.images
        assert input_images is not None
        input_batch = input_images.shape[0]
        input_size = input_images.shape[-2:]
        pred = outputs.get(self.segmentation_key, None)
        if pred is None:
            if self.segmentation_requires_pred:
                msg = f"Panoptic segmentation output not found in {outputs=}"
                raise RuntimeError(msg)
            pred = torch.full(
                (input_batch, 1, *input_size),
                PanopticMap.IGNORE,
                device=input_images.device,
                dtype=torch.int,
            ).as_subclass(PanopticMap)
        if pred.ndim == 4:
            pred.squeeze(1)
        assert pred.ndim == 3, f"Expected 3D tensor, got {pred.shape=}"
        assert pred.shape[-2:] == input_size, f"{pred.shape=} {input_size=}"
        assert pred.shape[0] == input_batch, f"{pred.shape=} {input_batch=}"

        if self.segmentation_task == SegmentationTask.SEMANTIC_SEGMENTATION:
            # Maybe convert the prediction to a PanopticMap if it is not already.
            # We use a simple heuristic to determine if the output is not encoded as PanopticMap,
            # see below:
            pred_max = pred.max()
            if pred_max < PanopticMap.DIVISOR or pred_max == 0:
                pred = PanopticMap.from_parts(pred, torch.zeros_like(pred))

        true = inputs.captures.segmentations
        if true is None:
            if self.segmentation_requires_true:
                msg = f"Panoptic segmentation target not found in {inputs=}"
                raise RuntimeError(msg)
            true = torch.full_like(pred, PanopticMap.IGNORE, dtype=torch.long)
        else:
            true = true[:, self.pair_index, ...]
        if true.ndim == 4:
            true.squeeze(1)
        assert true.ndim == 3, f"Expected 3D tensor, got {true.shape=}"
        assert true.shape[-2:] == input_size, f"{true.shape=} {input_size=}"
        assert true.shape[0] == input_batch, f"{true.shape=} {input_batch=}"

        valid = (true >= 0).any(dim=(-1)).any(dim=-1)

        for key, item in {
            self.segmentation_key_pred: pred,
            self.segmentation_key_true: true,
            self.segmentation_key_valid: valid,
        }.items():
            storage.set(key, item, inplace=True)

    @TX.override
    def _plot(self, storage: TensorDictBase, **kwargs) -> dict[str, pil_image.Image]:
        from unipercept.render import draw_image_segmentation

        result = super()._plot(storage, **kwargs)

        plot_keys = []
        plot_mapping = {
            self.segmentation_key_true: "segmentation_plot_true",
            self.segmentation_key_pred: "segmentation_plot_pred",
        }
        for key, mode_attr in plot_mapping.items():
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            if mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        for i in range(self.segmentation_plot_samples):
            for key in plot_keys:
                result[f"{key}_{i}"] = draw_image_segmentation(
                    storage.get_at(key, i), self.info
                )
        return result

    @TX.override
    def _compute(
        self, storage: TensorDictBase, **kwargs: T.Unpack[EvaluatorComputeKWArgs]
    ):
        return super()._compute(storage, **kwargs)


class COCOWriter(SegmentationWriter):
    @TX.override
    def _compute(
        self,
        storage: TensorDictBase,
        *,
        path: file_io.Path,
        device: torch.types.Device,
        **kwargs,
    ):
        results = super().compute(storage, path=path, device=device, **kwargs)

        path_files = file_io.Path(path) / "coco_panoptic"
        path_json = path_files.with_suffix(".json")

        assert not path_files.exists(), f"Path {path_files} already exists"
        assert not path_json.exists(), f"Path {path_json} already exists"

        self.logger.info("Exporting COCO panoptic segmentation to %s", path_json)

        path_files.mkdir(parents=True, exist_ok=True)

        coco_res: list[COCOResultPanoptic] = []
        sample_amt = len(storage)
        export_at = functools.partial(
            _export_coco_at,
            storage=storage,
            translations_dataset=self.info.translations_dataset,
            path_files=path_files,
        )
        progress_bar = self._progress_bar(
            total=sample_amt, desc="Exporting COCO panoptic"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            for result in pool.map(
                export_at,
                range(sample_amt),
            ):
                progress_bar.update(1)
                if result is not None:
                    coco_res.append(result)

        with open(path_json, "w") as f:
            json.dump(coco_res, f)

        return results

    @staticmethod
    def _export_coco_at(
        i: int,
        *,
        storage,
        translations_dataset,
        path_files,
        key_segmentation_valid,
        key_segmentation_true,
    ) -> COCOResultPanoptic | None:
        valid = storage.get_at(key_segmentation_valid, i).item()
        if not valid:
            return None
        true = T.cast(Tensor, storage.get_at(key_segmentation_true, i))
        true = true.as_subclass(PanopticMap)
        true.translate_semantic_(translations_dataset, inverse=True)
        coco_img, coco_info = true.to_coco()
        path_file = (path_files / str(i)).with_suffix(".png")
        coco_img.save(path_file, format="PNG")
        return {
            "image_id": i,
            "file_name": path_file.name,
            "segments_info": coco_info,
        }


@D.dataclass(kw_only=True)
class SemanticSegmentationEvaluator(SegmentationWriter):
    """
    Evaluates semantic segmentation metrics.

    The following metrics are computed:
    - Pixel accuracy (PA)
    - Mean intersection over union (mIoU)
    - Class-wise intersection over union (IoU)
    - Dice coefficient
    """

    # Override default values from parent classes
    show_details: bool = True
    segmentation_task: SegmentationTask = SegmentationTask.SEMANTIC_SEGMENTATION

    # Configurations specific for this evaluator
    semseg_with_details: bool = D.field(
        default=False,
        metadata={
            "help": "Show detailed metrics (i.e. for each category)",
        },
    )

    @TX.override
    def _compute(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, T.Any]:
        #  device = torch.device("cpu")  # TODO: GPU?

        num_cats = self.info.semantic_amount
        sample_amt = len(storage)
        assert sample_amt > 0, sample_amt
        miou_cls = torch.zeros(num_cats, dtype=torch.float32, device=device)
        miou_all = torch.zeros((1,), dtype=torch.float32, device=device)
        miou_samples = torch.zeros((1,), dtype=torch.int, device=device)
        compute_at = functools.partial(
            self._compute_semantic_metrics_at_index,
            storage=storage,
            device=device,
            num_categories=num_cats,
            key_segmentation_true=self.segmentation_key_true,
            key_segmentation_pred=self.segmentation_key_pred,
            key_segmentation_valid=self.segmentation_key_valid,
        )
        work = list(range(sample_amt))
        with (
            self._progress_bar(
                desc="Semantic segmentation metrics",
                total=sample_amt,
            ) as progress_bar,
            state.split_between_processes(work) as work_split,
            concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool,
        ):
            for result in pool.map(compute_at, work_split):
                progress_bar.update(1)
                if result is None:
                    continue
                miou_cls += result[0]
                miou_all += result[1]
                miou_samples += 1

        state.barrier()

        miou_samples = state.reduce(miou_samples, dst=0).wait()
        miou_cls = state.reduce(miou_cls, dst=0).wait() / miou_samples
        miou_all = state.reduce(miou_all, dst=0).wait() / miou_samples

        if not state.check_main_process():
            return {}

        miou_cls /= miou_samples
        miou_all /= miou_samples

        results = super().compute(storage, **kwargs)
        results["semseg.mIoU.all"] = miou_all.item()

        if self.show_summary:
            self.logger.info("Semantic segmentation evaluation: mIoU = %.2f", miou_all)
        if self.show_details:
            cat_name_list = [
                self.info.semantic_classes[k].name
                for k in sorted(self.info.semantic_classes.keys())
                if k >= 0
            ]
            cat_miou_list = miou_cls.tolist()
            assert len(cat_name_list) == len(
                cat_miou_list
            ), f"{cat_name_list=} {cat_miou_list=}"
            df = pd.DataFrame(
                {
                    "Category": cat_name_list,
                    "mIoU": cat_miou_list,
                }
            )
            self._show_table("Semantic segmentation details", df)
        if self.semseg_with_details:
            for i, semcls in self.info.semantic_classes.items():
                results[f"semseg.mIoU.{semcls.name}"] = miou_cls[i].item()

        return results

    @staticmethod
    def _compute_semantic_metrics_at_index(
        i,
        *,
        storage: TensorDictBase,
        device: torch.types.Device,
        num_categories: int,
        key_segmentation_true: str,
        key_segmentation_pred: str,
        key_segmentation_valid: str,
    ):
        valid = storage.get_at(key_segmentation_valid, i)
        if not valid:
            return None
        pred = storage.get_at(key_segmentation_pred, i).to(
            device=device, non_blocking=True
        )
        true = storage.get_at(key_segmentation_true, i).to(
            device=device, non_blocking=True
        )
        inter, union = compute_semantic_miou_partial(true, pred, num_categories)
        return accumulate_semantic_miou_partial(inter, union)


@torch.no_grad()
def accumulate_semantic_miou_partial(
    inter: PanopticMapLike, union: PanopticMapLike
) -> tuple[Tensor, Tensor]:
    """
    Accumulate the intersection and union tensors for mIoU calculation.

    See :func:`segmentation_miou_update_sample`, which outputs the inter and union samples.
    """
    miou = nonzero_divide(inter, union)
    return miou, miou.mean()


@torch.no_grad()
def compute_semantic_miou_partial(
    true: PanopticMapLike, pred: PanopticMapLike, n_cats: int
) -> tuple[Tensor, Tensor]:
    """
    Single sample update for mIoU calculation.
    """
    assert (true <= 0).all() or (true >= PanopticMap.DIVISOR).any()
    assert (pred <= 0).all() or (pred >= PanopticMap.DIVISOR).any()

    # Discard ignore category that is present in the ground truth
    ignore_mask = true < 0
    true = true[~ignore_mask]
    pred = pred[~ignore_mask]

    # Set negative values to new void category that is positive in the predictions
    pred[pred < 0] = n_cats
    n_cats += 1

    pred = PanopticMap.get_semantic_map(pred).long().flatten()
    assert pred.max() < n_cats, f"{pred.max()=} > {n_cats}"
    true = PanopticMap.get_semantic_map(true).long().flatten()
    assert true.max() < n_cats, f"{true.max()=} > {n_cats}"

    pred = nn.functional.one_hot(pred, num_classes=n_cats)[..., :-1].permute(1, 0)
    true = nn.functional.one_hot(true, num_classes=n_cats)[..., :-1].permute(1, 0)

    inter = (pred & true).sum(dim=1)
    union = true.sum(dim=1) + pred.sum(dim=1) - inter
    assert (inter <= union).all(), f"{inter=} {union=}"

    return inter, union


@torch.no_grad()
def compute_semantic_miou(
    true: PanopticMapLike | list[PanopticMapLike],
    pred: PanopticMapLike | list[PanopticMapLike],
    num_categories: int,
) -> tuple[Tensor, Tensor]:
    """
    Perform update and accumulation over multiple samples for mIoU calculation.
    """
    if isinstance(true, Tensor):
        assert isinstance(pred, Tensor)
        inter, union = compute_semantic_miou_partial(true, pred, num_categories)
        miou_per_cat, miou_all = accumulate_semantic_miou_partial(inter, union)
    elif isinstance(true, list):
        assert isinstance(pred, list)
        miou_per_cat = torch.zeros(
            num_categories, dtype=torch.long, device=true[0].device
        )
        miou_all = torch.zeros_like(miou_per_cat)
        miou_samples = 0
        for t, p in zip(true, pred, strict=False):
            inter, union = compute_semantic_miou_partial(t, p, num_categories)
            miou_per_cat_sample, miou_all_sample = accumulate_semantic_miou_partial(
                inter, union
            )
            miou_per_cat += miou_per_cat_sample
            miou_all += miou_all_sample
            miou_samples += 1
        miou_per_cat /= miou_samples
        miou_all /= miou_samples
    else:
        msg = f"Expected {true=} and {pred=} to be either list or tensor"
        raise TypeError(msg)
    return miou_per_cat, miou_all


# A (category_id, instance_id) tuple that uniquely identifies a panoptic segment.
_ColorType: T.TypeAlias = tuple[int, int]


class PQMetrics(T.NamedTuple):
    pq: Tensor
    sq: Tensor
    rq: Tensor
    iou: Tensor
    tp: Tensor
    fp: Tensor
    fn: Tensor


class PQDefinition(E.IntEnum):
    ORIGINAL = E.auto()
    BALANCED = E.auto()


@D.dataclass(kw_only=True)
class PanopticSegmentationEvaluator(SegmentationWriter):
    """
    Computes PQ metrics for panoptic segmentation tasks.


    See Also
    --------
    - Official `reference implementation <https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py>`_
    """

    panseg_with_details: bool = D.field(
        default=False,
        metadata={
            "help": "Show detailed metrics (i.e. for each category)",
        },
    )
    panseg_definition: PQDefinition = D.field(
        default=PQDefinition.ORIGINAL,
        metadata={
            "help": "The definition of the PQ metric to compute. Can be 'original' or 'balanced'.",
        },
    )

    @property
    def object_ids(self) -> frozenset[int]:
        return self.info.object_ids

    @property
    def background_ids(self) -> frozenset[int]:
        return self.info.background_ids

    @TX.override
    def _compute(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        metrics = super()._compute(storage, **kwargs)

        if self.panseg_definition & PQDefinition.ORIGINAL:
            metrics["original"] = self._compute_panoptic_quality(
                storage, device=kwargs["device"], allow_stuff_instances=True
            )
        if self.panseg_definition & PQDefinition.BALANCED:
            metrics["balanced"] = self._compute_panoptic_quality(
                storage, device=kwargs["device"], allow_stuff_instances=False
            )

        if len(metrics) == 0:
            raise ValueError("No PQ definition selected.")

        return metrics

    def _compute_panoptic_quality(
        self,
        storage: TensorDictBase,
        *,
        device: torch.types.Device,
        allow_stuff_instances: bool = False,
        allow_unknown_category: bool = True,
    ) -> dict[str, T.Any]:
        """
        Calculate stat scores required to compute the metric for a full batch.
        """
        void_color = find_void_color(self.object_ids, self.background_ids)
        # device = torch.device("cpu")  # using multiprocessing
        num_categories = len(self.object_ids) + len(self.background_ids)
        iou = torch.zeros(num_categories, dtype=torch.double, device=device)  # type: ignore
        tp = torch.zeros(num_categories, dtype=torch.int, device=device)  # type: ignore
        fp = torch.zeros_like(iou)
        fn = torch.zeros_like(fp)
        sample_amt = len(storage)
        assert (
            sample_amt > 0
        ), f"Batch size must be greater than zero, got {sample_amt=}"
        compute_at = functools.partial(
            self._compute_panoptic_quality_at_index,
            storage=storage,
            object_ids=self.object_ids,
            background_ids=self.background_ids,
            device=device,
            allow_unknown_category=allow_unknown_category,
            void_color=void_color,
            allow_stuff_instances=allow_stuff_instances,
            num_categories=num_categories,
            key_segmentation_true=self.segmentation_key_true,
            key_segmentation_pred=self.segmentation_key_pred,
            key_segmentation_valid=self.segmentation_key_valid,
        )

        work = range(sample_amt)
        with (
            self._progress_bar(
                desc="Panoptic segmentation metrics",
                total=sample_amt,
            ) as progress_bar,
            state.split_between_processes(work) as work_split,
            concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool,
        ):
            for result in pool.map(compute_at, work_split):
                progress_bar.update(1)
                if result is None:
                    continue
                iou += result[0]
                tp += result[1]
                fp += result[2]
                fn += result[3]

        state.barrier()

        iou, tp, fp, fn = (
            work.wait()
            for work in [
                state.reduce(iou),
                state.reduce(tp),
                state.reduce(fp),
                state.reduce(fn),
            ]
        )

        if not state.check_main_process():
            return {}

        # Compute PQ = SQ * RQ
        sq = nonzero_divide(iou, tp)
        rq = nonzero_divide(tp, tp + 0.5 * fp + 0.5 * fn)
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

        if self.panseg_with_details:
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
        tn_mask: Tensor = n_valid > 0
        th_mask: Tensor = isin(
            torch.arange(num_categories, device=metrics.pq.device),
            list(self.object_ids),
        )
        st_mask: Tensor = isin(
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
                    "PQ": 0.0,
                    # "SQ": 0.0,
                    # "RQ": 0.0,
                    # "IoU": 0.0,
                    # "TP": 0.0,
                    # "FP": 0.0,
                    # "FN": 0.0,
                }
            else:
                summary[name] = {
                    "PQ": metrics.pq[mask].mean().item(),
                    # "SQ": metrics.sq[mask].mean().item(),
                    # "RQ": metrics.rq[mask].mean().item(),
                    # "IoU": metrics.iou[mask].mean().item(),
                    # "TP": metrics.tp[mask].sum().item() / n_masked,
                    # "FP": metrics.fp[mask].sum().item() / n_masked,
                    # "FN": metrics.fn[mask].sum().item() / n_masked,
                }
        if self.show_summary:
            df = self._tabulate(summary)
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
                    "PQ": 0.0,
                    # "SQ": 0.0,
                    # "RQ": 0.0,
                    # "IoU": 0.0,
                    # "TP": 0.0,
                    # "FP": 0.0,
                    # "FN": 0.0,
                }
            else:
                details[name] = {
                    "PQ": metrics.pq[i].mean().item(),
                    # "SQ": metrics.sq[i].mean().item(),
                    # "RQ": metrics.rq[i].mean().item(),
                    # "IoU": metrics.iou[i].mean().item(),
                    # "TP": metrics.tp[i].sum().item() / n_masked,
                    # "FP": metrics.fp[i].sum().item() / n_masked,
                    # "FN": metrics.fn[i].sum().item() / n_masked,
                }
        if self.show_details:
            df = self._tabulate(details)
            msg = f"Panoptic details ({allow_stuff_instances=}, {allow_unknown_category=})"
            self._show_table(msg, df)
        return details

    @staticmethod
    def _compute_panoptic_quality_at_index(
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
        key_segmentation_true: str,
        key_segmentation_pred: str,
        key_segmentation_valid: str,
    ):
        valid = storage.get_at(key_segmentation_valid, n)
        if not valid:
            return None
        pred = storage.get_at(key_segmentation_pred, n).to(
            device=device, non_blocking=True
        )
        true = storage.get_at(key_segmentation_true, n).to(
            device=device, non_blocking=True
        )
        pred = preprocess_panoptic_quality(
            object_ids,
            background_ids,
            pred,
            void_color=void_color,
            allow_unknown_category=allow_unknown_category,
        )
        true = preprocess_panoptic_quality(
            object_ids,
            background_ids,
            true,
            void_color=void_color,
            allow_unknown_category=True,
        )
        result = compute_panoptic_quality_partial(
            pred,
            true,
            void_color=void_color,
            background_ids=background_ids if not allow_stuff_instances else None,
            num_categories=num_categories,
        )
        return result

    @staticmethod
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


def _to_tuple(t: Tensor) -> tuple:
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


def _get_color_areas(inputs: Tensor) -> dict[tuple, Tensor]:
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
    return dict(zip(_to_tuple(unique_keys), unique_keys_area, strict=False))


def find_void_color(
    thing_colors: frozenset[int], stuff_colors: frozenset[int]
) -> tuple[int, int]:
    r"""
    Suggest an unused color ID for ignored/void segments.

    In the canonical implementation, the void color cannot be negative, so this value
    is mapped to the maximum of the category IDs plus one.

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
    unused_category_id = 1 + max([0, *list(thing_colors), *list(stuff_colors)])
    return unused_category_id, 0


def preprocess_panoptic_quality(
    things: frozenset[int],
    stuffs: frozenset[int],
    inputs: PanopticMap,
    void_color: tuple[int, int],
    allow_unknown_category: bool,
) -> Tensor:
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

    mask_stuffs = isin(out[:, 0], list(stuffs))
    mask_things = isin(out[:, 0], list(things))

    if not allow_unknown_category and not torch.all(mask_things | mask_stuffs):
        raise ValueError(
            f"Unknown categories found: {out[~(mask_things|mask_stuffs)].unique().cpu().tolist()}"
        )

    # Set unknown categories to void color
    out[~(mask_things | mask_stuffs)] = out.new(void_color)

    return out


def compute_panoptic_quality_partial(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    void_color: tuple[int, int],
    num_categories: int,
    background_ids: frozenset[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Calculate stat scores required to compute the metric for a single sample.
    """

    def _compute_iou(
        pred_color: _ColorType,
        target_color: _ColorType,
        pred_areas: dict[_ColorType, Tensor],
        target_areas: dict[_ColorType, Tensor],
        intersection_areas: dict[tuple[_ColorType, _ColorType], Tensor],
    ) -> Tensor:
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
        union = (
            pred_area - pred_void_area + target_area - void_target_area - intersection
        )
        return intersection / union

    def _filter_fn(
        target_areas: dict[_ColorType, Tensor],
        target_segment_matched: set[_ColorType],
        intersection_areas: dict[tuple[_ColorType, _ColorType], Tensor],
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

    def _filter_fp(
        pred_areas: dict[_ColorType, Tensor],
        pred_segment_matched: set[_ColorType],
        intersection_areas: dict[tuple[_ColorType, _ColorType], Tensor],
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

    background_ids = background_ids or frozenset()
    device = flatten_preds.device

    iou_sum = torch.zeros(num_categories, dtype=torch.double, device=device)
    tp = torch.zeros(num_categories, dtype=torch.int, device=device)
    fp = torch.zeros(num_categories, dtype=torch.int, device=device)
    fn = torch.zeros(num_categories, dtype=torch.int, device=device)

    # Calculate the area of each prediction, ground truth and pairwise intersection.
    pred_areas = T.cast(dict[_ColorType, Tensor], _get_color_areas(flatten_preds))
    tgt_areas = T.cast(dict[_ColorType, Tensor], _get_color_areas(flatten_target))
    # intersection matrix of shape [num_pixels, 2, 2]
    inter_matrix = torch.transpose(
        torch.stack((flatten_preds, flatten_target), -1), -1, -2
    )
    assert inter_matrix.shape == (flatten_preds.shape[0], 2, 2)
    inter_areas = T.cast(
        dict[tuple[_ColorType, _ColorType], Tensor],
        _get_color_areas(inter_matrix),
    )

    # select intersection of things of same category with iou > 0.5
    pred_segment_matched = set()
    tgt_seg_matched = set()
    for pred_color, target_color in inter_areas.keys():
        # test only non void, matching category
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        iou = _compute_iou(
            pred_color,
            target_color,
            pred_areas,
            tgt_areas,
            inter_areas,
        )
        sem_id = target_color[0]
        if target_color[0] not in background_ids and iou > 0.5:
            pred_segment_matched.add(pred_color)
            tgt_seg_matched.add(target_color)
            iou_sum[sem_id] += iou
            tp[sem_id] += 1
        elif target_color[0] in background_ids and iou > 0:
            iou_sum[sem_id] += iou

    for cat_id in _filter_fn(tgt_areas, tgt_seg_matched, inter_areas):
        if cat_id not in background_ids:
            fn[cat_id] += 1

    for cat_id in _filter_fp(pred_areas, pred_segment_matched, inter_areas):
        if cat_id not in background_ids:
            fp[cat_id] += 1

    for cat_id, _ in tgt_areas:
        if cat_id in background_ids:
            tp[cat_id] += 1

    return iou_sum, tp, fp, fn
