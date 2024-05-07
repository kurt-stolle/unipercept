"""
Implements an evaluator for depth estimation tasks.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses as D
import functools
import typing as T

import torch
import torch.multiprocessing as M
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase
from torch import Tensor
from torch.utils._pytree import tree_map
from tqdm import tqdm
from typing_extensions import override

from unipercept.log import create_table, get_logger
from unipercept.state import check_main_process, cpus_available

from ._base import Evaluator, PlotMode

if T.TYPE_CHECKING:
    from ..data.sets import Metadata

_logger = get_logger(__name__)

__all__ = [
    "DepthEvaluator",
    "DepthWriter",
    "DepthMetrics",
    "PRED_DEPTH",
    "TRUE_DEPTH",
    "VALID_DEPTH",
    "compute_depth_metrics",
]

PRED_DEPTH: T.Final[str] = "pred_depth"
TRUE_DEPTH: T.Final[str] = "true_depth"
VALID_DEPTH: T.Final[str] = "valid_depth"

_FLOAT_DTYPE: T.Final = torch.float64
_FLOAT_EPS: T.Final = torch.finfo(_FLOAT_DTYPE).eps


@D.dataclass(kw_only=True)
class DepthWriter(Evaluator):
    """
    Writes depth maps to storage for evaluation purposes.
    """

    info: Metadata = D.field(repr=False)

    plot_samples: int = 1
    plot_true: PlotMode = PlotMode.ONCE
    plot_pred: PlotMode = PlotMode.ALWAYS

    pred_key = "depths"
    true_key = ("captures", "depths")
    true_group_index = -1  # the most recent group, assuming temporal ordering

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        """
        This method is a stub for a ``from_metadata`` classmethod that would use the metadata of a dataset to
        instantiate this evaluator.
        """
        from unipercept import get_info

        info = get_info(name)
        return cls(info=info, **kwargs)

    @override
    def update(
        self, storage: TensorDictBase, inputs: TensorDictBase, outputs: TensorDictBase
    ):
        super().update(storage, inputs, outputs)

        storage_keys = storage.keys(leaves_only=True, include_nested=True)
        if (
            TRUE_DEPTH in storage_keys
            and PRED_DEPTH in storage_keys
            and VALID_DEPTH in storage_keys
        ):
            return

        pred = outputs.get(self.pred_key, None)
        assert pred.dtype == torch.float32
        if pred is None:
            raise RuntimeError(f"Missing key {self.pred_key} in {outputs=}")

        true = inputs.get(self.true_key, None)
        assert true.dtype == torch.float32
        if true is None:  # Generate dummy values for robust evaluation downstream
            true = torch.full_like(pred, 0, dtype=torch.float32)
        else:
            true = true[:, self.true_group_index, ...]

        assert (
            true.ndim == 3
        ), f"Expected 3D tensor for {self.true_key}, got {true.shape=}"

        valid = (true > 1e-8).any(-1).any(-1)

        for key, item in ((TRUE_DEPTH, true), (PRED_DEPTH, pred), (VALID_DEPTH, valid)):
            storage.set(key, item, inplace=True)

    @override
    def compute(self, *args, **kwargs):
        return {}

    @override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        from unipercept.render import draw_image_depth

        plot_keys = []
        for key, mode_attr in ((TRUE_DEPTH, "plot_true"), (PRED_DEPTH, "plot_pred")):
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            elif mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        result = {}
        for i in range(self.plot_samples):
            for key in plot_keys:
                result[f"{key}_{i}"] = draw_image_depth(
                    storage.get_at(key, i).clone().float(), self.info
                )
        return result


_KEY_VALID_PX = "valid"


@D.dataclass(kw_only=True)
class DepthEvaluator(DepthWriter):
    show_progress: bool = True

    show_summary: bool = True

    @classmethod
    @override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @override
    def compute(self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs):
        num_samples = storage.batch_size[0]
        compute_at = functools.partial(self._compute_at, storage=storage, device=device)
        metrics_list: list[DepthMetrics] = []

        progress_bar = tqdm(
            total=num_samples,
            desc="Computing depth metrics",
            disable=not check_main_process(local=True) or not self.show_progress,
        )
        for metrics_sample in map(compute_at, range(num_samples)):
            progress_bar.update(1)
            if metrics_sample is None:
                continue
            metrics_list.append(metrics_sample)
        progress_bar.close()

        # Compute the final metrics as the average of the samples weighted by the amount of valid pixels at each entry
        metrics = accumulate_partial_depth_metrics(metrics_list)
        metrics = tree_map(lambda x: x.item(), metrics._asdict())

        if self.show_summary:
            _logger.info("Depth summary:\n%s", create_table(metrics, format="wide"))

        # Add metrics from parent class
        metrics.update(super().compute(storage, device=device))

        return metrics

    @staticmethod
    def _compute_at(n, *, storage, device):
        valid = storage.get_at(VALID_DEPTH, n).item()
        if not valid:
            return None

        pred = storage.get_at(PRED_DEPTH, n, None).to(device=device)
        true = storage.get_at(TRUE_DEPTH, n, None).to(device=device)

        if pred is None or true is None:
            return None

        return compute_partial_depth_metrics(pred=pred, true=true)


class DepthMetrics(T.NamedTuple):
    """
    Metrics for depth estimation tasks.
    """

    valid: Tensor
    abs_rel: Tensor
    sq_rel: Tensor
    rmse: Tensor
    rmse_log: Tensor
    accuracy: dict[str, Tensor]


_THRES_DEFAULT: T.Final[list[int]] = [1, 2, 3]


def _threshold_to_key(t_base: float, n: int) -> str:
    """
    Converts a threshold value (float) to a string key for the accuracy dict.

    Parameters
    ----------
    t_base : float
        Threshold value base (e.g. 1.25).
    n : int
        Threshold value exponent (e.g. 2 for 1.25**2).

    Returns
    -------
    str
        String key for the accuracy dict, e.g. "1t25**2" for a threshold of 1.25**2.
    """

    base = f"{t_base}".replace(".", "t")
    exponent = f"**{n}"

    return f"{base}{exponent}"


def _get_valid_depths(
    pred: Tensor, true: Tensor, threshold: float = 1.0
) -> T.Tuple[Tensor, Tensor, Tensor]:
    """
    Returns a mask for valid pixels in the ground truth depth map.

    Parameters
    ----------
    pred : Tensor
        Predicted depth map.
    true : Tensor
        Ground truth depth map.
    threshold : float, optional
        Minimal value for valid pixels in the ground truth depth map, by default 1.0.

    Returns
    -------
    Tensor
        Mask for valid pixels in the ground truth depth map.
    """

    mask = true >= threshold
    amount = mask.to(dtype=torch.int64).sum()

    return pred[mask], true[mask], amount


def _align_and_promote(pred: Tensor, true: Tensor):
    r"""
    Aligns and promotes the input tensors to accurate floating-point tensors.
    """
    if not torch.is_floating_point(pred):
        msg = f"Expected floating-point tensor for prediction, got {pred_dtype=}"
        raise TypeError(msg)
    if not torch.is_floating_point(true):
        msg = f"Expected floating-point tensor for ground truth, got {true_dtype=}"
        raise TypeError(msg)
    pred_dtype = pred.dtype
    true_dtype = true.dtype
    #pred = pred.to(dtype=pred_dtype).to(dtype=_FLOAT_DTYPE)
    #true = true.to(dtype=true_dtype).to(dtype=_FLOAT_DTYPE)

    pred = pred.to(dtype=_FLOAT_DTYPE)
    true = true.to(dtype=_FLOAT_DTYPE)

    return pred, true


@torch.no_grad()
def compute_depth_metrics(
    *,
    pred: Tensor,
    true: Tensor,
    t_base: float = 1.25,
    t_n: T.Iterable[int] = _THRES_DEFAULT,
) -> DepthMetrics:
    """
    Computation of error metrics between predicted and ground truth depths.

    Parameters
    ----------
    pred : Tensor
        Predicted depth map.
    true : Tensor
        Ground truth depth map.
    t_base : float, optional
        Base value for the accuracy thresholds, by default 1.25.
    t_n : T.Iterable[int], optional
        Exponents for the accuracy thresholds, by default [1, 2, 3].

    Returns
    -------
    DepthMetrics
        The computed metrics.
    """

    pred, true = map(torch.flatten, (pred, true))
    pred, true = _align_and_promote(pred, true)
    pred, true, px_amt = _get_valid_depths(pred, true)

    max_rel = torch.maximum((true / pred), (pred / true))

    return DepthMetrics(
        valid=px_amt,
        abs_rel=((true - pred).abs() / true).mean(),
        sq_rel=((true - pred).square() / true).mean(),
        rmse=(true - pred).square().mean().sqrt(),
        rmse_log=((torch.log(true) - torch.log(pred)) ** 2).mean().sqrt(),
        accuracy={
            _threshold_to_key(t_base, n): (max_rel < (t_base**n)).double().mean()
            for n in t_n
        },
    )


@torch.no_grad()
def compute_partial_depth_metrics(
    *,
    pred: Tensor,
    true: Tensor,
    t_base: float = 1.25,
    t_n: T.Iterable[int] = _THRES_DEFAULT,
) -> DepthMetrics:
    """
    Computation of error metrics between predicted and ground truth depths.

    Parameters
    ----------
    pred : Tensor
        Predicted depth map.
    true : Tensor
        Ground truth depth map.
    t_base : float, optional
        Base value for the accuracy thresholds, by default 1.25.
    t_n : T.Iterable[int], optional
        Exponents for the accuracy thresholds, by default [1, 2, 3].

    Returns
    -------
    DepthMetrics | None
        The partially computed metrics, which still need to be accumulated.
    """

    pred, true = map(torch.flatten, (pred, true))
    pred, true = _align_and_promote(pred, true)
    pred, true, px_amt = _get_valid_depths(pred, true)
    max_rel = torch.maximum((true / pred), (pred / true))
    return DepthMetrics(
        valid=px_amt,
        abs_rel=((true - pred).abs_() / true).sum(),
        sq_rel=((true - pred).square_() / true).sum(),
        rmse=((true - pred) ** 2).sum(),
        rmse_log=((torch.log1p(true) - torch.log1p(pred)) ** 2).sum(),
        accuracy={
            _threshold_to_key(t_base, n): (max_rel < (t_base**n)).long().sum()
            for n in t_n
        },
    )


@torch.no_grad()
def accumulate_partial_depth_metrics(
    metrics: T.Iterable[DepthMetrics], *, device: torch.device | None = None
):
    r"""
    Accumulates the partial depth metrics into a single set of metrics.
    """
    if device is None:
        device = next(iter(metrics)).valid.device
    else:
        device = torch.device(device)

    accuracy_keys = list(next(iter(metrics)).accuracy.keys())
    metrics = [m for m in metrics if m.valid > 0]

    with device:
        size1 = (1,)
        if len(metrics) == 0:
            return DepthMetrics(
                valid=torch.zeros(size1, dtype=torch.int64),
                abs_rel=torch.full(size1, fill_value=torch.inf, dtype=_FLOAT_DTYPE),
                sq_rel=torch.full(size1, fill_value=torch.inf, dtype=_FLOAT_DTYPE),
                rmse=torch.full(size1, fill_value=torch.inf, dtype=_FLOAT_DTYPE),
                rmse_log=torch.full(size1, fill_value=torch.inf, dtype=_FLOAT_DTYPE),
                accuracy={
                    key: torch.full(size1, fill_value=torch.nan, dtype=_FLOAT_DTYPE)
                    for key in next(iter(metrics)).accuracy.keys()
                },
            )

        # Allocate tensors for the accumulated metrics
        valid = torch.zeros(1, dtype=torch.int64)
        abs_rel = torch.zeros(1, dtype=_FLOAT_DTYPE)
        sq_rel = torch.zeros(1, dtype=_FLOAT_DTYPE)
        rmse = torch.zeros(1, dtype=_FLOAT_DTYPE)
        rmse_log = torch.zeros(1, dtype=_FLOAT_DTYPE)
        accuracy = {key: torch.zeros(1, dtype=_FLOAT_DTYPE) for key in accuracy_keys}

    # Accumulate the metrics
    for metric in metrics:
        valid += metric.valid.to(device=device)
        abs_rel += metric.abs_rel.to(device=device)
        sq_rel += metric.sq_rel.to(device=device)
        rmse += metric.rmse.to(device=device)
        rmse_log += metric.rmse_log.to(device=device)
        for key in accuracy_keys:
            accuracy[key] += metric.accuracy[key].to(device=device)

    # Find the average metrics and finish the computation
    return DepthMetrics(
        valid=valid,
        abs_rel=abs_rel / valid,
        sq_rel=sq_rel / valid,
        rmse=(rmse / valid).sqrt_(),
        rmse_log=(rmse_log / valid).sqrt_(),
        accuracy={key: value / valid for key, value in accuracy.items()},
    )
