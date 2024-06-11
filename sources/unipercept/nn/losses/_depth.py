# from __future__ import annotations

from __future__ import annotations

import typing as T
import typing_extensions as TX
import enum as E

import torch
from torch import nn, Tensor

from unipercept.nn.losses.mixins import ScaledLossMixin

__all__ = [
    "RMSELoss",
    "SILogLoss",
    "PEDLoss",
    "IGSLoss",
    "SSILoss",
    "compute_igs_loss",
    "RMSELoss",
    "GradientLoss",
]

#######################
# Utilities for depth #
#######################


def _reduce_batch(image_loss, M):
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def _reduce_pixels(image_loss, M):
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def _mean_var_with_mask(
    data: Tensor,
    mask: Tensor,
    dim: T.Tuple[int, ...] | int | str,
    keepdim: bool = False,
) -> T.Tuple[Tensor, Tensor]:
    r"""
    Compute the mean and variance of the data tensor along the specified dimension, optionally
    using a mask tensor.
    """
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    if not keepdim:
        mask_mean = mask_mean.squeeze(dim)
        mask_var = mask_var.squeeze(dim)

    return mask_mean, mask_var


def _mean_with_mask(
    data: Tensor,
    mask: Tensor | None,
    dim: T.Tuple[int, ...] | int | str,
    keepdim: bool = False,
) -> Tensor:
    r"""
    Compute the mean of the data tensor along the specified dimension, optionally
    using a mask tensor.
    """
    if mask is None:
        return data.mean(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    if not keepdim:
        mask_mean = mask_mean.squeeze(dim)
    return mask_mean


def _median_with_mask(
    data: Tensor, mask: Tensor | None, dim: int | str, keepdim: bool = False
) -> Tensor:
    r"""
    Compute the median of the data tensor along the specified dimension, optionally
    using a mask tensor.
    """
    if mask is None:
        return data.median(dim=dim, keepdim=keepdim).values
    data = torch.where(mask, data, torch.nan)
    return data.nanmedian(dim=dim, keepdim=keepdim).values


def _median_mad_with_mask(
    data: Tensor, mask: Tensor | None, dim: int | str, keepdim: bool = False
) -> T.Tuple[Tensor, Tensor]:
    """
    Compute the median and mean absolute deviation of the data tensor along the specified
    dimension, optionally using a mask tensor.
    """
    if mask is None:
        median = data.median(dim=dim, keepdim=keepdim).values
        mad = (data - median).abs().median(dim=dim, keepdim=keepdim).values
        return median, mad
    data = torch.where(mask, data, torch.nan)
    median = data.nanmedian(dim=dim, keepdim=True).values
    mad = (data - median).abs().nanmedian(dim=dim, keepdim=True).values
    if not keepdim:
        median = median.squeeze(dim)
        mad = mad.squeeze(dim)
    return median, mad


####################################
# Scale-Invariant Logarithmic Loss #
####################################


def compute_silog_loss(
    src: Tensor,
    tgt: Tensor,
    mask: Tensor,
    dim: T.Tuple[int, ...],
    alpha: float = 0.15,
    eps: float = 1e-5,
):
    r"""
    Compute the Scale-Invariant Logarithmic (SILog) loss between the source and target.

    Parameters
    ----------
    src : Tensor[N, ..., *dim]
        The source tensor.
    tgt : Tensor[N, ..., *dim] or Tensor[N, ..., *dim, P]
        The target tensor. If the target tensor has an additional dimension P, then a
        margin of error is computed using the median absolute deviation over the P dimension.
    mask : Tensor[N, ..., *dim]
        The mask tensor.
    dim : Tuple[int, ...]
        The dimensions to reduce over.
    eps : float
        The epsilon value to clamp the error to.
    """
    src = src.float()
    tgt = tgt.float()

    if src.ndim != tgt.ndim:
        assert tgt.ndim == src.ndim + 1, (tgt.shape, src.shape)
        tgt, tgt_mad = _median_mad_with_mask(tgt, tgt > 0, -1)
        tgt[~mask] = 0.0
        tgt_mad[~mask] = 0.0

        src_log = src.clamp(eps).log()
        err_min = src_log - (tgt - tgt_mad).clamp(eps).log()
        err_max = src_log - (tgt + tgt_mad).clamp(eps).log()
        err = torch.min(err_min.abs(), err_max.abs())
    else:
        err = src.clamp(min=eps).log() - tgt.clamp(min=eps).log()

    # Compute the error
    err_mean, err_var = _mean_var_with_mask(err, mask, dim, keepdim=True)
    err_scale = err_mean**2

    if err_var.ndim > 1:
        err_var = err_var.sum(dim=1)
        err_scale = err_scale.sum(dim=1)
    err_scale = alpha * err_scale
    return (err_var + err_scale).clamp_min(eps).sqrt()


class SILogLoss(ScaledLossMixin, nn.Module):
    alpha: T.Final[float]
    dim: T.Final[T.Tuple[int, ...]]
    eps: T.Final[float]

    def __init__(
        self,
        alpha: float = 0.15,
        eps: float = 1e-5,
        dim: T.Tuple[int, ...] = (-2, -1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.dim = dim
        self.eps: float = eps

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""
        See :func:`compute_silog_loss` for more details.
        """
        loss = compute_silog_loss(input, target, mask, self.dim, self.alpha, self.eps)
        return loss * self.scale


def compute_rmse_loss(
    src: Tensor,
    tgt: Tensor,
    mask: Tensor,
    dim: T.Tuple[int, ...],
    eps: float = 1e-6,
):
    r"""
    Compute the Root Mean Squared Error (RMSE) loss between the source and target
    tensors.

    Parameters
    ----------
    src : Tensor[N, ..., *dim]
        The source tensor.
    tgt : Tensor[N, ..., *dim] or Tensor[N, ..., *dim, P]
        The target tensor. If the target tensor has an additional dimension P, then a
        margin of error is computed using the median absolute deviation over the P dimension.
    mask : Tensor[N, ..., *dim]
        The mask tensor.
    dim : Tuple[int, ...]
        The dimensions to reduce over.
    eps : float
        The epsilon value to clamp the error to.
    """
    src = src.float()
    tgt = tgt.float()

    if src.ndim != tgt.ndim:
        assert tgt.ndim == src.ndim + 1, (tgt.shape, src.shape)
        tgt, tgt_mad = _median_mad_with_mask(tgt, tgt > 0, -1)
        tgt[~mask] = 0.0
        tgt_mad[~mask] = 0.0
    else:
        tgt_mad = torch.zeros_like(tgt)

    err = ((src - tgt).abs() - tgt_mad).clamp_min(eps)
    err = _mean_with_mask(data=err.square(), mask=mask, dim=dim).sqrt()
    if err.ndim > 1:
        err = err.sum(dim=1)
    return err


class RMSELoss(ScaledLossMixin, nn.Module):
    dim: T.Final[T.Tuple[int, ...]]
    eps: T.Final[float]

    def __init__(
        self,
        dim: T.Tuple[int, ...] = (-2, -1),
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    @classmethod
    def from_metadata(
        cls, dataset_name: str, init_scale: float = 1.0, **kwargs
    ) -> T.Self:
        r"""
        Initialize from metadata, fills out the scale parameter using the range of
        depth values expected in the dataset such that the loss is nomalized.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to use for initializing the loss.
        init_scale : float
            The scale of initial values of predictions centered around the mean depth
            value. By default 1.0.
        **kwargs : dict
            Additional keyword arguments to pass to the loss.
        """
        from unipercept.data.sets import catalog, Metadata

        info: Metadata = catalog.get_info(dataset_name)
        assert info.depth_max is not None
        assert info.depth_min is not None

        # The scale is computed by taking `n_points` in the range of depth values,
        # computing the RMSE, and then scaling the loss such that the RMSE is 1.0.
        scale = kwargs.pop("scale", 1.0)
        n_points = 1000
        with torch.no_grad():
            tgt = torch.linspace(-1 / 2, 1 / 2, n_points, dtype=torch.float)
            src = tgt.flip(0) * init_scale

            dst_range = info.depth_max - info.depth_min
            dst_mean = dst_range / 2.0

            tgt = tgt * dst_range + dst_mean
            src = src * dst_range + dst_mean

            rmse = (tgt - tgt.flip(0) * init_scale).square().mean().sqrt()
            scale *= (1.0 / rmse).item()
            del rmse
            del tgt

        return cls(scale=scale, **kwargs)

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        See :func:`compute_rmse_loss` for more details.
        """
        err = compute_rmse_loss(src, tgt, mask, self.dim, self.eps)
        return err * self.scale


class PEDLoss(nn.Module):
    """
    Panoptic-guided Edge Discontinuity Loss (PED) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self):
        super().__init__()

    @TX.override
    def forward(
        self, output: Tensor, target: Tensor
    ):  # NOTE:  target is panoptic mask, output is norm disparity
        output = output.float()
        target = target.float()

        # Compute the Iverson bracket for adjacent pixels along the x-dimension
        panoptic_diff_x = torch.diff(target, dim=-1) != 0

        # Compute the Iverson bracket for adjacent pixels along the y-dimension
        panoptic_diff_y = torch.diff(target, dim=-2) != 0

        # Compute the partial disp derivative along the x-axis
        disp_diff_x = torch.diff(output, dim=-1)

        # Compute the partial disp derivative along the y-axis
        disp_diff_y = torch.diff(output, dim=-2)

        loss = torch.mean(
            torch.mul(panoptic_diff_x, torch.exp(-torch.abs(disp_diff_x)))
        ) + torch.mean(torch.mul(panoptic_diff_y, torch.exp(-torch.abs(disp_diff_y))))

        return loss


def compute_igs_loss(pred_disparity, true_image):
    """
    Compute the smoothness loss for a given disparity map.

    Parameters
    ----------
    output
        Predicted disparity map.
    target
        Input image (RGB or grayscale)
    """

    # Compute the Iverson bracket for adjacent pixels along the x-dimension
    image_diff_x = torch.mean(torch.diff(true_image, dim=-1), dim=1, keepdim=True)
    # Compute the Iverson bracket for adjacent pixels along the y-dimension
    image_diff_y = torch.mean(torch.diff(true_image, dim=-2), dim=1, keepdim=True)

    # Compute the partial disp derivative along the x-axis
    disp_diff_x = torch.diff(pred_disparity, dim=-1)

    # Compute the partial disp derivative along the y-axis
    disp_diff_y = torch.diff(pred_disparity, dim=-2)

    loss = torch.mean(
        torch.mul(torch.abs(disp_diff_x), torch.exp(-torch.abs(image_diff_x)))
    ) + torch.mean(
        torch.mul(torch.abs(disp_diff_y), torch.exp(-torch.abs(image_diff_y)))
    )

    return loss


class IGSLoss(nn.Module):
    r"""
    Image-guided smoothness loss (IGS) loss
    """

    def __init__(self):
        super().__init__()

    @TX.override
    def forward(self, output, target):
        return compute_igs_loss(output, target)


def compute_gradient_loss(prediction, target, mask):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return _reduce_batch(image_loss, M)


class GradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()

        self.__scales = scales

    @TX.override
    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        total = []

        for scale in range(self.__scales):
            step = pow(2, scale)

            total.append(
                compute_gradient_loss(
                    prediction[:, ::step, ::step],
                    target[:, ::step, ::step],
                    mask[:, ::step, ::step],
                )
            )

        return torch.stack(total).sum()


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class SSILoss(nn.Module):
    """
    Scale and Shift Invariant (SSI) loss
    """

    def __init__(self, alpha=0.5, scales=4):
        super().__init__()

        self.rmse = RMSELoss()
        self.gradient = GradientLoss(scales=scales)
        self.alpha = alpha

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(self, prediction, target, mask):
        prediction = prediction.float()
        target = target.float()
        mask = mask.bool()

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.rmse(ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.gradient(ssi, target, mask)
        return total
