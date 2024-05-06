# from __future__ import annotations

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.nn.losses.mixins import ScaledLossMixin, StableLossMixin
from unipercept.utils.mask import masks_to_boxes

__all__ = [
    "DepthLoss",
    "DepthSmoothLoss",
    "PEDLoss",
    "SimpleSmoothnessLoss",
    "ScaleAndShiftInvariantLoss",
    "compute_smoothness_loss",
    "MSELoss",
    "GradientLoss",
]


def scale_invariant_logarithmic_error(
    x: torch.Tensor, y: torch.Tensor, num: int, eps: float
) -> torch.Tensor:
    r"""
    Scale invariant logarithmic error.
    """
    log_err = x.log() - y.log()
    sile_1 = torch.mean(log_err**2)
    sile_2 = (torch.sum(log_err) ** 2) / (num * num)
    return sile_1 - sile_2


def relative_absolute_squared_error(
    x: torch.Tensor, y: torch.Tensor, num: int, eps: float
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square relative error and absolute relative error.
    """
    err = x - y
    err_rel = err / y
    are = err_rel.abs().sum() / num

    sre = err_rel.square().sum() / num
    sre = sre.clamp(eps).sqrt()

    return are, sre


def relative_sqrt_error(
    x: torch.Tensor, y: torch.Tensor, num: int, eps: float
) -> torch.Tensor:
    r"""
    Computes relative square root error.
    """
    return torch.sqrt(torch.mean((1 - x / y) ** 2).clamp(eps))


class DepthLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    def __init__(
        self,
        *,
        weight_sile=5.0,
        weight_arse=0.0,
        weight_srse=0.0,
        weight_rsqe=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_sile = weight_sile
        self.weight_arse = weight_arse
        self.weight_srse = weight_srse
        self.weight_rsqe = weight_rsqe

    @override
    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert mask.dtype == torch.bool, mask.dtype
        if not mask.any():
            return pred.sum() * 0.0
        true = true[mask].float().clamp(self.eps)
        pred = pred[mask].float().clamp(self.eps)
        n = true.numel()

        if self.weight_arse > 0 or self.weight_srse > 0:
            are, sre = self._compute_arse(pred, true, n)

            if self.weight_arse > 0:
                are = are * self.weight_arse
            else:
                are = None

            if self.weight_srse > 0:
                sre = sre * self.weight_srse
            else:
                sre = None
        else:
            are = sre = None

        if self.weight_sile > 0:
            sile = self._compute_sile(pred, true, n) * self.weight_sile
        else:
            sile = None

        if self.weight_rsqe > 0:
            rsqe = self._compute_rsqe(pred, true, n) * self.weight_rsqe
        else:
            rsqe = None

        loss_list = [sile, rsqe, are, sre]
        loss = torch.stack(
            [loss_comp for loss_comp in loss_list if loss_comp is not None]
        )

        return loss.sum() * self.scale

    def _compute_sile(
        self, pred: torch.Tensor, true: torch.Tensor, num: int
    ) -> torch.Tensor:
        return scale_invariant_logarithmic_error(pred, true, num, self.eps)

    def _compute_arse(
        self, pred: torch.Tensor, true: torch.Tensor, num: int
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        return relative_absolute_squared_error(pred, true, num, self.eps)

    def _compute_rsqe(
        self, pred: torch.Tensor, true: torch.Tensor, num: int
    ) -> torch.Tensor:
        return relative_sqrt_error(pred, true, num, self.eps)


class DepthSmoothLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    """
    Compute the depth smoothness loss, defined as the weighted smoothness
    of the inverse depth.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padded = False

    @override
    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        mask: torch.Optional[torch.Tensor],
    ) -> torch.Tensor:
        images = images.float()
        depths = depths.float()

        if len(images) == 0:
            return depths.sum()

        if mask is None:
            mask = torch.ones_like(images).bool()
        else:
            mask = mask.bool()

        # compute inverse depths
        # idepths = 1 / (depths / self.depth_max).clamp(self.eps)
        # idepths = depths.float() / self.depth_max
        # idepths = depths
        idepths = 1 / depths.clamp(self.eps)

        # compute the gradients
        idepth_dx: torch.Tensor = self._gradient_x(idepths)
        idepth_dy: torch.Tensor = self._gradient_y(idepths)
        image_dx: torch.Tensor = self._gradient_x(images)
        image_dy: torch.Tensor = self._gradient_y(images)

        # compute image weights
        weights_x: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dx) + self.eps, dim=1, keepdim=True)
        )
        weights_y: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dy) + self.eps, dim=1, keepdim=True)
        )

        # apply image weights to depth
        smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

        # compute shifted masks
        if self.padded:
            h, w = mask.shape[-2], mask.shape[-1]

            mask_boxes = masks_to_boxes(mask.squeeze(1).long())

            assert all(mask_boxes[:, 2] < w)
            assert all(mask_boxes[:, 3] < h)

            mask_margin = 1 / 10
            mask_boxes[:, 0] = (mask_boxes[:, 0] - int(w * mask_margin)).clamp(min=0)
            mask_boxes[:, 1] = (mask_boxes[:, 1] - int(h * mask_margin)).clamp(min=0)
            mask_boxes[:, 2] = (mask_boxes[:, 2] + int(w * mask_margin)).clamp(max=w)
            mask_boxes[:, 3] = (mask_boxes[:, 3] + int(h * mask_margin)).clamp(max=h)

            mask_padded = torch.full_like(mask, False)
            for i, (x_min, y_min, x_max, y_max) in enumerate(mask_boxes):
                mask_padded[i, 0, y_min:y_max, x_min:x_max] = True

            mask_x = mask_padded[:, :, :, 1:]
            mask_y = mask_padded[:, :, 1:, :]
        else:
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]

        # loss for x and y
        loss_x = smoothness_x[mask_x].mean()
        loss_y = smoothness_y[mask_y].mean()

        assert loss_x.isfinite()
        assert loss_y.isfinite()

        return loss_x + loss_y

    def _gradient_x(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _gradient_y(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] - img[:, :, 1:, :]


class PEDLoss(nn.Module):
    """
    Panoptic-guided Edge Discontinuity Loss (PED) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self):
        super().__init__()

    @override
    def forward(
        self, output: torch.Tensor, target: torch.Tensor
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


def compute_smoothness_loss(output, target):
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
    image_diff_x = torch.mean(torch.diff(target, dim=-1), dim=1, keepdim=True)
    # Compute the Iverson bracket for adjacent pixels along the y-dimension
    image_diff_y = torch.mean(torch.diff(target, dim=-2), dim=1, keepdim=True)

    # Compute the partial disp derivative along the x-axis
    disp_diff_x = torch.diff(output, dim=-1)

    # Compute the partial disp derivative along the y-axis
    disp_diff_y = torch.diff(output, dim=-2)

    loss = torch.mean(
        torch.mul(torch.abs(disp_diff_x), torch.exp(-torch.abs(image_diff_x)))
    ) + torch.mean(
        torch.mul(torch.abs(disp_diff_y), torch.exp(-torch.abs(image_diff_y)))
    )

    return loss


class SimpleSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, output, target):
        return compute_smoothness_loss(output, target)


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


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
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

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.mse = MSELoss(reduction=reduction)
        self.gradient = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha
        self.ssi = None

    @override
    def forward(self, prediction, target, mask):
        prediction = prediction.float()
        target = target.float()
        mask = mask.bool()

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.mse(self.ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.gradient(self.ssi, target, mask)
        return total
