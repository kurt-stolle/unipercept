"""
Implements an evaluator for depth estimation tasks.

Computes the metrics proposed by Eigen et al. (2014) for depth estimation tasks.
"""

from __future__ import annotations

import dataclasses as D
import functools
import typing as T

import torch
import torch.types
import typing_extensions as TX
from tensordict import TensorDictBase
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

import unipercept.log as up_log
from unipercept import state

from ._base import Evaluator, PlotMode, StoragePrefix

if T.TYPE_CHECKING:
    from ..model import InputData

_logger = up_log.get_logger(__name__)

_FLOAT_DTYPE: T.Final = torch.float64
_FLOAT_EPS: T.Final = torch.finfo(_FLOAT_DTYPE).eps


@D.dataclass(kw_only=True)
class DepthWriter(Evaluator):
    """
    Writes depth maps to storage for evaluation purposes.
    """

    depth_plot_samples: int | None = 1
    depth_plot_true: PlotMode = PlotMode.ONCE
    depth_plot_pred: PlotMode = PlotMode.ALWAYS
    depth_plot_error: PlotMode = PlotMode.ALWAYS
    depth_plot_confidence: PlotMode = PlotMode.ALWAYS
    depth_plot_uncertainty: PlotMode = PlotMode.ALWAYS

    depth_requires_true: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the target depth is not found"},
    )
    depth_requires_pred: bool = D.field(
        default=True,
        metadata={"help": "Raise an error if the predicted depth is not found"},
    )
    depth_requires_conf: bool = D.field(
        default=False,
        metadata={
            "help": "Raise an error if the predicted depth confidence map is not found"
        },
    )
    depth_requires_uncert: bool = D.field(
        default=False,
        metadata={
            "help": "Raise an error if the predicted depth uncertainty map is not found"
        },
    )
    depth_metric_key: str = D.field(
        default="depth", metadata={"help": "Key for the depth map"}
    )

    @property
    def depth_key_true(self):
        return self._get_storage_key(self.depth_metric_key, StoragePrefix.TRUE)

    @property
    def depth_key_pred(self):
        return self._get_storage_key(self.depth_metric_key, StoragePrefix.PRED)

    @property
    def depth_key_valid(self):
        return self._get_storage_key(self.depth_metric_key, StoragePrefix.VALID)

    @TX.override
    def _update(
        self,
        storage: TensorDictBase,
        inputs: InputData,
        outputs: TensorDictBase,
        **kwargs,
    ):
        super()._update(storage, inputs, outputs, **kwargs)

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

        pred = outputs.get(self.depth_metric_key, None)
        if pred is None:
            if self.depth_requires_pred:
                msg = f"Missing key {self.depth_metric_key} in {outputs.keys()=}"
                raise RuntimeError(msg)
            pred = torch.zeros(
                (input_batch, *input_shape),
                dtype=torch.float32,
                device=input_images.device,
            )
        assert pred.dtype == torch.float32, pred.dtype
        if pred.ndim == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        assert pred.ndim == 3, pred.shape
        assert pred.shape[-2:] == input_shape, (pred.shape, input_shape)
        assert pred.shape[0] == input_batch, (pred.shape, input_batch)
        true = inputs.captures.depths
        if true is None:
            if self.depth_requires_true:
                msg = f"Missing key {self.depth_metric_key} in {inputs.keys()=}"
                raise RuntimeError(msg)
            true = torch.full_like(pred, 0, dtype=torch.float32)
        else:
            assert isinstance(true, torch.Tensor), type(true)
            assert true.dtype == torch.float32
            true = true[:, self.pair_index, ...]
        if true.ndim == 4 and true.size(1) == 1:
            true = true.squeeze(1)
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
    def _compute(self, *args, **kwargs):
        return super()._compute(*args, **kwargs)

    def _plot_true_pred(self, storage: TensorDictBase) -> dict[str, T.Any]:
        from unipercept.render import draw_image_depth

        plot_keys = []
        for key, mode_attr in (
            (self.depth_key_true, "depth_plot_true"),
            (self.depth_key_pred, "depth_plot_pred"),
        ):
            mode = getattr(self, mode_attr)
            if mode == PlotMode.NEVER:
                continue
            if mode == PlotMode.ONCE:
                setattr(self, mode_attr, PlotMode.NEVER)
            plot_keys.append(key)

        result = {}
        for i in range(
            self.depth_plot_samples
            if self.depth_plot_samples is not None
            else len(storage)
        ):
            for key in plot_keys:
                val = storage.get_at(key, i, default=None)  # type: ignore
                if val is None:
                    continue
                val = val.clone().float()
                result[f"{key}_{i}"] = draw_image_depth(val, self.info)
        return result

    def _plot_error(self, storage: TensorDictBase) -> dict[str, T.Any]:
        from unipercept.render import plot_depth_error

        result = {}
        if self.depth_plot_error == PlotMode.NEVER:
            return result
        if self.depth_plot_error == PlotMode.ONCE:
            self.depth_plot_error = PlotMode.NEVER

        for i in range(self.depth_plot_samples):
            pred = storage.get_at(self.depth_key_pred, i).clone().float()
            true = storage.get_at(self.depth_key_true, i).clone().float()

            result[f"depth_plot_error_{i}"] = plot_depth_error(pred, true, self.info)
        return result

    @TX.override
    def _plot(self, storage: TensorDictBase, **kwargs) -> dict[str, T.Any]:
        result = super()._plot(storage, **kwargs)
        result.update(self._plot_true_pred(storage))
        result.update(self._plot_error(storage))
        return result


@D.dataclass(kw_only=True)
class DepthEvaluator(DepthWriter):
    depth_per_sample: bool = D.field(
        default=True,
        metadata={
            "help": (
                "When set to True, the evaluator computes metrics for each sample and "
                "report the average over all samples. "
                "If False, the evaluator computes the metrics for the entire dataset, "
                "as if all samples were concatenated."
            )
        },
    )

    @TX.override
    def compute(self, storage: TensorDictBase, **kwargs):
        metrics = super().compute(storage, **kwargs)
        metrics.update(self._compute_eigen_metrics(storage, **kwargs))

        return metrics

    def _compute_eigen_metrics(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, T.Any]:
        num_samples = len(storage)
        compute_at = functools.partial(
            self._compute_depth_metrics_at_index, storage=storage, device=device
        )
        metrics_list: list[EigenMetrics] = []
        metrics_count = torch.zeros((1,), device=device, dtype=torch.int64)
        work = list(range(num_samples))
        with (
            state.split_between_processes(work) as work_split,
            self._progress_bar(
                total=num_samples, desc="Computing depth metrics"
            ) as progress_bar,
        ):
            for metrics_sample in map(compute_at, work_split):
                progress_bar.update(1)
                if metrics_sample is None:
                    continue
                metrics_list.append(metrics_sample)
                metrics_count += 1

        # Ensure that all processes have the same number of samples
        metrics = concat_eigen_metrics(metrics_list)
        metrics = tree_map(lambda x: x.sum(), metrics)

        # Synchronize the processes
        state.barrier()

        # Reduce between processes
        metrics_count = state.reduce(metrics_count).wait()
        metrics = tree_map(lambda x: state.reduce(x).wait(), metrics)

        if not state.check_main_process():
            return {}

        # Accumulate
        if self.depth_per_sample:
            metrics_dict = tree_map(
                lambda x: (x / metrics_count).item(), metrics._asdict()
            )
        else:
            metrics_dict = accumulate_eigen_partial(metrics)
            metrics_dict = tree_map(lambda x: x.item(), metrics._asdict())
        metrics_dict = T.cast(dict, metrics_dict)

        if self.show_summary:
            msg = "Depth metrics (Eigen et al.)"
            self._show_table(msg, metrics_dict)

        return metrics_dict

    def _compute_depth_metrics_at_index(
        self, n, *, storage, device
    ) -> EigenMetrics | None:
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

        if self.depth_per_sample:
            fn = compute_eigen_metrics
        else:
            fn = compute_eigen_partial
        return fn(pred=pred, true=true)


class EigenMetrics(T.NamedTuple):
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
    pred: Tensor, true: Tensor, reject_outliers: bool = True, threshold: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns a mask for valid pixels in the ground truth depth map.

    Parameters
    ----------
    pred : Tensor
        Predicted depth map.
    true : Tensor
        Ground truth depth map.
    reject_outliers: bool, optional
        Reject pixels where the absolute relative error is an outlier, by default True.
    threshold : float, optional
        Minimal value for valid pixels in the ground truth depth map, by default 1.0.

    Returns
    -------
    Tensor
        Mask for valid pixels in the ground truth depth map.
    """

    mask = true >= threshold
    true = true[mask]
    pred = pred[mask]

    if reject_outliers:
        err = (true.log1p() - pred.log1p()).abs()
        err_mean = err.mean()
        err_std = err.std()
        mask = (err - err_mean).abs() <= 3.0 * err_std
        true = true[mask]
        pred = pred[mask]

    amount = mask.to(dtype=torch.int64).sum()

    return pred, true, amount


def _align_and_promote(pred: Tensor, true: Tensor):
    r"""
    Aligns and promotes the input tensors to accurate floating-point tensors.
    """
    if not torch.is_floating_point(pred):
        msg = f"Expected floating-point tensor for prediction, got {pred.dtype=}"
        raise TypeError(msg)
    if not torch.is_floating_point(true):
        msg = f"Expected floating-point tensor for ground truth, got {true.dtype=}"
        raise TypeError(msg)
    pred = pred.to(dtype=_FLOAT_DTYPE)
    true = true.to(dtype=_FLOAT_DTYPE)
    return pred, true


def _trunc_to_uint16(depth: Tensor) -> Tensor:
    r"""
    Many benchmarks are based on 16-bit PNG depth maps. Since models outputs' are
    floats, we need to compress them to match the precision of the benchmarks.
    """
    if torch.is_floating_point(depth):
        depth = depth.to(_FLOAT_DTYPE) * 256.0
        depth = depth.clamp(0, 2**16 - 1)
        depth = depth / 256.0
    return depth


############################################
# Eigen metrics for per-sample computation #
############################################


def compute_eigen_metrics(
    *,
    pred: Tensor,
    true: Tensor,
    t_base: float = 1.25,
    t_n: T.Iterable[int] = _THRES_DEFAULT,
    trunc_to_uint16: bool = True,
    reject_outliers: bool = True,
) -> EigenMetrics:
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
    trunc_to_uint16: bool, optional
        Truncate the depth maps (floating point) to 16-bit unsigned integers, by default True.
    reject_outliers: bool, optional
        Reject outliers in the depth maps, by default True.

    Returns
    -------
    DepthMetrics
        The computed metrics.
    """

    assert pred.shape == true.shape, (pred.shape, true.shape)
    assert pred.ndim in (2, 3), pred.shape

    pred, true, px_amt = _get_valid_depths(pred, true, reject_outliers)

    if trunc_to_uint16:
        true, pred = map(_trunc_to_uint16, (true, pred))

    if pred.ndim == 3:
        metrics = []
        for i in range(pred.shape[0]):
            metrics.append(
                compute_eigen_metrics(
                    pred=pred[i], true=true[i], t_base=t_base, t_n=t_n
                )
            )

        metrics_concat = concat_eigen_metrics(metrics)
        return tree_map(lambda x: x.sum() / pred.shape[0], metrics_concat)

    pred, true = map(torch.flatten, (pred, true))
    pred, true = _align_and_promote(pred, true)

    max_rel = torch.maximum((true / pred), (pred / true))
    err = true - pred
    err_log = true.log() - pred.log()

    return EigenMetrics(
        valid=px_amt,
        abs_rel=((err).abs() / true).mean(),
        sq_rel=((err).square() / true.square()).mean(),
        rmse=(err).square().mean().sqrt(),
        rmse_log=err_log.square().mean().sqrt(),
        accuracy={
            _threshold_to_key(t_base, n): (max_rel < (t_base**n)).double().mean()
            for n in t_n
        },
    )


def concat_eigen_metrics(metrics: T.Iterable[EigenMetrics]) -> EigenMetrics:
    """
    Concatenates the depth metrics computed for each sample.
    """
    metrics = tuple(metrics)
    if len(metrics) == 0:
        msg = "No depth metrics to concatenate"
        raise ValueError(msg)

    assert all(isinstance(m, EigenMetrics) for m in metrics)

    _, spec = tree_flatten(metrics[0])
    tensors, _ = tree_flatten(metrics)
    tensors = [torch.atleast_1d(t) for t in tensors]
    tensors_per_sample = len(tensors) // len(metrics)

    tensors_concat = [
        torch.cat(tensors[i::tensors_per_sample]) for i in range(tensors_per_sample)
    ]

    return tree_unflatten(tensors_concat, spec)


##############################################
# Partial variant, allows global computation #
##############################################


def compute_eigen_partial(
    *,
    pred: Tensor,
    true: Tensor,
    t_base: float = 1.25,
    t_n: T.Iterable[int] = _THRES_DEFAULT,
    trunc_to_uint16: bool = True,
    reject_outliers: bool = True,
) -> EigenMetrics:
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
    trunc_to_uint16: bool, optional
        Truncate the depth maps (floating point) to 16-bit unsigned integers, by default True.
    reject_outliers: bool, optional
        Reject outliers in the depth maps, by default True.

    Returns
    -------
    DepthMetrics | None
        The partially computed metrics, which still need to be accumulated.
    """

    pred, true, px_amt = _get_valid_depths(pred, true, reject_outliers)
    if trunc_to_uint16:
        true, pred = map(_trunc_to_uint16, (true, pred))

    pred, true = map(torch.flatten, (pred, true))
    pred, true = _align_and_promote(pred, true)
    max_rel = torch.maximum((true / pred), (pred / true))
    return EigenMetrics(
        valid=px_amt,
        abs_rel=((true - pred).abs_() / true).sum(),
        sq_rel=((true - pred).square_() / true).sum(),
        rmse=((true - pred) ** 2).sum(),
        rmse_log=(torch.log(true) - torch.log(pred)).square().sum(),
        accuracy={
            _threshold_to_key(t_base, n): (max_rel < (t_base**n)).long().sum()
            for n in t_n
        },
    )


def accumulate_eigen_partial(metrics: EigenMetrics | T.Iterable[EigenMetrics]):
    r"""
    Accumulates the partial depth metrics into a single set of metrics.
    """
    if isinstance(metrics, EigenMetrics):
        valid = metrics.valid.sum()
        return EigenMetrics(
            valid=metrics.valid.sum() / valid.numel(),
            abs_rel=metrics.abs_rel.sum() / valid,
            sq_rel=metrics.sq_rel.sum() / valid,
            rmse=(metrics.rmse.sum() / valid).sqrt_(),
            rmse_log=(metrics.rmse_log.sum() / valid).sqrt_(),
            accuracy={
                key: value.sum() / valid for key, value in metrics.accuracy.items()
            },
        )
    if isinstance(metrics, T.Iterable):
        metrics = concat_eigen_metrics(metrics)
        return accumulate_eigen_partial(metrics)
    msg = f"Expected iterable or instance of EigenMetrics, got {type(metrics)=}"
    raise TypeError(msg)
