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
from tqdm import tqdm
from typing_extensions import override

from unipercept.state import check_main_process, cpus_available

from ._base import Evaluator, PlotMode

if T.TYPE_CHECKING:
    from ..data.sets import Metadata

__all__ = [
    "DepthEvaluator",
    "DepthWriter",
    "DepthMetrics",
    "PRED_DEPTH",
    "TRUE_DEPTH",
    "VALID_DEPTH",
]

PRED_DEPTH: T.Final[str] = "pred_depth"
TRUE_DEPTH: T.Final[str] = "true_depth"
VALID_DEPTH: T.Final[str] = "valid_depth"


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
        if pred is None:
            raise RuntimeError(f"Missing key {self.pred_key} in {outputs=}")

        true = inputs.get(self.true_key, None)
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

        result = super().plot(storage)
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

    @classmethod
    @override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @override
    def compute(self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs):
        # TODO
        device = torch.device("cpu")
        num_samples = storage.batch_size[0]
        assert num_samples > 0
        compute_at = functools.partial(_compute_at, storage=storage, device=device)
        metrics_list: list[DepthMetrics] = []
        progress_bar = tqdm(
            total=num_samples,
            desc="Computing depth metrics",
            disable=not check_main_process(local=True) or not self.show_progress,
        )
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for metrics_sample in pool.map(compute_at, range(num_samples)):
                progress_bar.update(1)
                if metrics_sample is None:
                    continue
                metrics_list.append(metrics_sample)
        progress_bar.close()

        # Compute the final metrics as the average of the samples weighted by the amount of valid pixels at each entry
        metrics = {}
        for m in metrics_list:  # accumulate metrics
            valid_pixels = m[_KEY_VALID_PX]
            metrics[_KEY_VALID_PX] = metrics.get(_KEY_VALID_PX, 0) + valid_pixels
            for k, v in m.items():
                if k == _KEY_VALID_PX:
                    continue
                elif k == "accuracy":
                    assert isinstance(v, dict)
                    metrics.setdefault(k, {th: 0.0 for th in v.keys()})
                    for i in v.keys():
                        metrics[k][i] += v[i] * valid_pixels
                else:
                    assert isinstance(v, float)
                    metrics.setdefault(k, 0.0)
                    metrics[k] += v * valid_pixels
        for k, v in metrics.items():  # divide by total pixels
            if k == _KEY_VALID_PX:
                continue
            elif k == "accuracy":
                assert isinstance(v, dict)
                for i in v.keys():
                    v[i] /= metrics[_KEY_VALID_PX]
            else:
                assert isinstance(v, float)
                v /= metrics[_KEY_VALID_PX]

        # Add metrics from parent class
        metrics.update(super().compute(storage, device=device))

        return metrics


def _compute_at(n, *, storage, device):
    valid = storage.get_at(VALID_DEPTH, n).item()
    if not valid:
        return None

    pred = storage.get_at(PRED_DEPTH, n, None).to(device=device)
    true = storage.get_at(TRUE_DEPTH, n, None).to(device=device)

    if pred is None or true is None:
        return None

    return _depth_metrics_single(pred=pred, true=true)


DepthMetrics = T.TypedDict(
    "DepthMetrics",
    {
        "valid": int,
        "abs_rel": float,
        "sq_rel": float,
        "rmse": float,
        "log_rmse": float,
        "accuracy": dict[str, float],
    },
)


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


def _depth_metrics_single(
    *,
    pred: torch.Tensor,
    true: torch.Tensor,
    t_base: float = 1.25,
    t_n: T.Iterable[int] = _THRES_DEFAULT,
    eps=1e-8,
) -> DepthMetrics | None:
    """
    Computation of error metrics between predicted and ground truth depths.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted depth map.
    true : torch.Tensor
        Ground truth depth map.
    t_base : float, optional
        Base value for the accuracy thresholds, by default 1.25.
    t_n : T.Iterable[int], optional
        Exponents for the accuracy thresholds, by default [1, 2, 3].
    eps : float, optional
        Epsilon value used to ensure numeric stability, by default 1e-8.

    Returns
    -------
    DepthMetrics | None
        Dictionary with the computed metrics or None if the truth contains no valid depths.
    """

    pred = pred.flatten()
    true = true.flatten()

    # Mask out invalid pixels
    valid_mask = true > eps
    valid_amt = int(valid_mask.short().sum().item())

    if valid_amt <= 0:
        return None
    pred = pred[valid_mask].double().clamp(min=eps)
    true = true[valid_mask].double().clamp(min=eps)

    # Compute thresholds
    max_rel = torch.maximum((true / pred), (pred / true))

    # Mean accuracies at different thresholds
    accuracy = {
        _threshold_to_key(t_base, n): (max_rel < (t_base**n)).double().mean()
        for n in t_n
    }

    rmse = (true - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log1p(true) - torch.log1p(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(true - pred) / true)

    sq_rel = torch.mean(((true - pred) ** 2) / true)

    return {
        _KEY_VALID_PX: valid_amt,
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "accuracy": {k: a.item() for k, a in accuracy.items()},
    }
