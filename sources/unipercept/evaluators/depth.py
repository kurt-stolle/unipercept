"""
Implements an evaluator for depth estimation tasks.

Computes the metrics proposed by Eigen et al. (2014) for depth estimation tasks.
"""

from __future__ import annotations

import typing_extensions as TX
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
from unipercept.state import check_main_process, cpus_available, get_interactive

from ._base import Evaluator, PlotMode, StoragePrefix

if T.TYPE_CHECKING:
    from ..data.sets import Metadata
    from ..model import InputData

_logger = get_logger(__name__)

_FLOAT_DTYPE: T.Final = torch.float64
_FLOAT_EPS: T.Final = torch.finfo(_FLOAT_DTYPE).eps


@D.dataclass(kw_only=True)
class DepthWriter(Evaluator):
    """
    Writes depth maps to storage for evaluation purposes.
    """

    depth_plot_samples: int = 1
    depth_plot_true: PlotMode = PlotMode.ONCE
    depth_plot_pred: PlotMode = PlotMode.ALWAYS
    depth_requires_true: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the target depth is not found"},
    )
    depth_requires_pred: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the predicted depth is not found"},
    )
    depth_key: str = D.field(
        default="depth", metadata={"help": "Key for the depth map"}
    )

    @property
    def depth_key_true(self):
        return self.get_storage_key(self.depth_key, StoragePrefix.TRUE)

    @property
    def depth_key_pred(self):
        return self.get_storage_key(self.depth_key, StoragePrefix.PRED)

    @property
    def depth_key_valid(self):
        return self.get_storage_key(self.depth_key, StoragePrefix.VALID)

    @TX.override
    def update(
        self, storage: TensorDictBase, inputs: InputData, outputs: TensorDictBase
    ):
        super().update(storage, inputs, outputs)

        target_keys = {
            self.depth_key_true,
            self.depth_key_pred,
            self.depth_key_valid,
        }
        storage_keys = storage.keys(leaves_only=True, include_nested=True)
        assert storage_keys is not None
        if target_keys.issubset(storage_keys):
            return

        input_images = inputs.captures.images
        assert input_images is not None
        input_batch = input_images.shape[0]
        input_shape = input_images.shape[-2:]

        pred = outputs.get(self.depth_key, None)
        if pred is None:
            if self.depth_requires_pred:
                msg = f"Missing key {self.depth_key} in {outputs.keys()=}"
                raise RuntimeError(msg)
            pred = torch.zeros(
                (input_batch, *input_shape),
                dtype=torch.float32,
                device=input_images.device,
            )
        assert pred.dtype == torch.float32, pred.dtype
        assert pred.ndim == 3, pred.shape
        assert pred.shape[-2:] == input_shape, (pred.shape, input_shape)
        assert pred.shape[0] == input_batch, (pred.shape, input_batch)

        true = inputs.captures.depths
        if true is None:
            if self.depth_requires_true:
                msg = f"Missing key {self.depth_key} in {inputs.keys()=}"
                raise RuntimeError(msg)
            true = torch.full_like(pred, 0, dtype=torch.float32)
        else:
            assert isinstance(true, torch.Tensor), type(true)
            assert true.dtype == torch.float32
            true = true[:, self.pair_index, ...]
        assert true.ndim == 3, true.shape
        assert true.shape[-2:] == input_shape, (true.shape, input_shape)
        assert true.shape[0] == input_batch, (true.shape, input_batch)

        valid = (true > 1e-8).any(-1).any(-1)

        for key, item in {
            self.depth_key_true: true,
            self.depth_key_pred: pred,
            self.depth_key_valid: valid,
        }.items():
            storage.set(key, item, inplace=True)

    @TX.override
    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs)

    @TX.override
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        from unipercept.render import draw_image_depth

        result = super().plot(storage)

        plot_keys = []
        for key, mode_attr in (
            (self.depth_key_true, "depth_plot_true"),
            (self.depth_key_pred, "depth_plot_pred"),
        ):
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            elif mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        for i in range(self.depth_plot_samples):
            for key in plot_keys:
                result[f"{key}_{i}"] = draw_image_depth(
                    storage.get_at(key, i).clone().float(), self.info
                )
        return result


@D.dataclass(kw_only=True)
class DepthEvaluator(DepthWriter):

    @classmethod
    @TX.override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @TX.override
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

    def _compute_at(self, n, *, storage, device):
        valid = storage.get_at(self.depth_key_valid, n).item()
        if not valid:
            return None
        pred = storage.get_at(self.depth_key_pred, n, None).to(
            device=device, non_blocking=True
        )
        true = storage.get_at(self.depth_key_true, n, None).to(
            device=device, non_blocking=True
        )
        assert pred is not None and true is not None, (pred, true)
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
    # pred = pred.to(dtype=pred_dtype).to(dtype=_FLOAT_DTYPE)
    # true = true.to(dtype=true_dtype).to(dtype=_FLOAT_DTYPE)

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
