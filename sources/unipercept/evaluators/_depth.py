"""
Implements an evaluator for depth estimation tasks.
"""
from __future__ import annotations

import dataclasses as D
import typing as T

import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

from ..data.tensors import DepthMap
from ..log import get_logger
from ._base import Evaluator, PlotMode

if T.TYPE_CHECKING:
    from ..data.sets import Metadata
_logger = get_logger(__name__)

PRED_DEPTH = "pred_depth"
TRUE_DEPTH = "true_depth"

KEY_VALID_PXS = "valid"


@D.dataclass(kw_only=True)
class DepthWriter(Evaluator):
    """
    Writes depth maps to storage for evaluation purposes.
    """

    info: Metadata = D.field(repr=False)

    plot_samples: int = 1
    plot_true: PlotMode = PlotMode.ONCE
    plot_pred: PlotMode = PlotMode.ALWAYS

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
    def update(self, storage: TensorDictBase, inputs: TensorDictBase, outputs: TensorDictBase):
        super().update(storage, inputs, outputs)
        storage.setdefault(TRUE_DEPTH, inputs.get("captures", "depths"), inplace=True)
        storage.setdefault(PRED_DEPTH, outputs.get("depths", None), inplace=True)

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
                result[f"{key}_{i}"] = draw_image_depth(storage.get_at(key, i).clone(), self.info)
        return result


class DepthEvaluator(DepthWriter):
    @classmethod
    @override
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        return super().from_metadata(name, **kwargs)

    @override
    def compute(
        self, storage: TensorDictBase, *, device: torch.types.Device, **kwargs
    ) -> dict[str, int | float | str | bool]:
        # TODO
        num_samples = storage.batch_size[0]
        assert num_samples > 0

        metrics_list: list[DepthMetrics] = []
        for n in range(num_samples):
            pred = storage.get_at(PRED_DEPTH, n, None).clone().to(device=device)
            true = storage.get_at(TRUE_DEPTH, n, None).clone().to(device=device)

            if pred is None or true is None:
                continue

            metrics_sample = _depth_metrics_single(pred=pred, true=true)
            if metrics_sample is None:
                continue
            else:
                metrics_list.append(metrics_sample)

        # Compute the final metrics as the average of the samples weighted by the amount of valid pixels at each entry
        metrics = {}

        for m in metrics_list:  # accumulate metrics
            valid_pixels = m[KEY_VALID_PXS]
            metrics[KEY_VALID_PXS] = metrics.get(KEY_VALID_PXS, 0) + valid_pixels
            for k, v in m.items():
                if k == KEY_VALID_PXS:
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
            if k == KEY_VALID_PXS:
                continue
            elif k == "accuracy":
                assert isinstance(v, dict)
                for i in v.keys():
                    v[i] /= metrics[KEY_VALID_PXS]
            else:
                assert isinstance(v, float)
                v /= metrics[KEY_VALID_PXS]

        # Add metrics from parent class
        metrics.update(super().compute(storage, device=device))

        return metrics


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


DEFAULT_THRESHOLDS: T.Final[list[int]] = [1, 2, 3]


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
    *, pred: torch.Tensor, true: torch.Tensor, t_base: float = 1.25, t_n: T.Iterable[int] = DEFAULT_THRESHOLDS, eps=1e-8
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
    accuracy = {_threshold_to_key(t_base, n): (max_rel < (t_base**n)).double().mean() for n in t_n}

    rmse = (true - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log1p(true) - torch.log1p(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(true - pred) / true)

    sq_rel = torch.mean(((true - pred) ** 2) / true)

    return {
        KEY_VALID_PXS: valid_amt,
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "accuracy": {k: a.item() for k, a in accuracy.items()},
    }
