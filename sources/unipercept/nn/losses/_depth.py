# from __future__ import annotations

from __future__ import annotations

import math
import typing as T
import typing_extensions as TX

import torch
from torch import nn, Tensor
from typing_extensions import override

from unipercept.nn.losses.mixins import ScaledLossMixin, StableLossMixin
from unipercept.utils.mask import masks_to_boxes

__all__ = [
    "MSELoss",
    "SILogLoss",
    "PEDLoss",
    "IGSLoss",
    "ScaleAndShiftInvariantLoss",
    "compute_igs_loss",
    "MSELoss",
    "GradientLoss",
]


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


def _masked_mean_var(data: Tensor, mask: Tensor, dim: T.Tuple[int, ...]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)


def _masked_mean(data: Tensor, mask: Tensor | None, dim: T.Tuple[int, ...]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


class SILogLoss(ScaledLossMixin, nn.Module):
    def __init__(
        self,
        alpha: float = 0.15,
        eps: float = 1e-5,
        dims: T.Tuple[int, ...] = (-2, -1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.dims = dims
        self.eps: float = eps

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        input = input.float()
        target = target.float()
        error = input.clamp(min=self.eps).log1p() - target.clamp(min=self.eps).log1p()
        mean_error, var_error = _masked_mean_var(error, mask, self.dims)
        scale_error = mean_error**2

        if var_error.ndim > 1:
            var_error = var_error.sum(dim=1)
            scale_error = scale_error.sum(dim=1)
        scale_error = self.alpha * scale_error
        loss = (
            (var_error + scale_error)
            .clamp_min(self.eps)
            .sqrt()
            .mean(dim=0, keepdim=True)
            .mean()
        )

        return loss * self.scale


class MSELoss(ScaledLossMixin, nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        dims: T.Tuple[int, ...] = (-2, -1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight: float = weight
        self.dims = dims
        self.eps = 1e-6

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        error = (input - target).square().sum(dim=self.dims)
        error = _masked_mean(data=error, mask=mask, dim=self.dims).mean(dim=self.dims)
        return error.mean() * self.scale


class PEDLoss(nn.Module):
    """
    Panoptic-guided Edge Discontinuity Loss (PED) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self):
        super().__init__()

    @override
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

    @override
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


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4):
        super().__init__()

        self.mse = MSELoss()
        self.gradient = GradientLoss(scales=scales)
        self.alpha = alpha

    @override
    @torch.autocast("cuda", enabled=False)
    def forward(self, prediction, target, mask):
        prediction = prediction.float()
        target = target.float()
        mask = mask.bool()

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.mse(ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.gradient(ssi, target, mask)
        return total
