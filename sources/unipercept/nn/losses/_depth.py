# from __future__ import annotations

from __future__ import annotations

import typing as T
import typing_extensions as TX
import enum as E

import torch
from torch import nn, Tensor

from unipercept.nn.losses.mixins import ScaledLossMixin

__all__ = [
    "MSELoss",
    "RelativeLoss",
    "SILogLoss",
    "PEDLoss",
    "IGSLoss",
    "SSILoss",
    "compute_igs_loss",
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


class AvgStat(E.StrEnum):
    r"""
    Enum for the average statistics available in this module.
    """

    MEAN = E.auto()
    MEDIAN = E.auto()
    MODE = E.auto()


class VarStat(E.StrEnum):
    r"""
    Enum for the variance statistics available in this module.
    """

    VAR = E.auto()
    MAD = E.auto()
    IQR = E.auto()


def _mean_var_with_mask(
    data: Tensor,
    mask: Tensor,
    *,
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
    *,
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
    data: Tensor, mask: Tensor | None, *, dim: int | str, keepdim: bool = False
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
    data: Tensor, mask: Tensor | None, *, dim: int | str, keepdim: bool = False
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


_DEFAULT_MARGIN_SCALE: T.Final[float] = 1.0


def _target_error_margin(
    avg: Tensor,
    mask: Tensor,
    *,
    scale: float = _DEFAULT_MARGIN_SCALE,
    eps: float = 1e-6,
) -> T.Tuple[Tensor, Tensor]:
    r"""
    Compute the error margin of the target. Assumes the target has a dimension containing
    the folded (i.e. stacked) pixels that would normally be merged during a downsampling
    operation. This function computes an average (e.g. median) and deviation (e.g. MAD)
    over the folded dimension to compute an error margin for the target.

    Parameters
    ----------
    tgt : Tensor[..., *dim, P]
        The target tensor. Only entries greater than 0 are considered valid.
    mask : Tensor[..., *dim]
        The mask tensor. Entries with `True` are considered valid.
    scale : float
        A parameter that controls the error margin. By default 1.0.
    eps : float
        The epsilon value to add to the average to prevent division by zero.

    Returns
    -------
    Tensor[..., *dim]
        The average target tensor.
    Tensor[..., *dim]
        The scaled margin of error.
    """

    # Compute average and deviation statistics
    avg, dev = _median_mad_with_mask(avg, avg > 0, dim=-1, keepdim=False)
    avg[~mask] = 0.0
    dev[~mask] = 0.0

    # Compute the allowed margin of error for the target from the average and deviation,
    # considering the relative deviation for each pixel.
    margin = dev / (avg + eps)
    margin = margin * scale

    return avg, margin


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
    margin_scale: float = _DEFAULT_MARGIN_SCALE,
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
    margin_scale : float
        The scale of the margin of error. See :func:`_target_error_margin` for more details.
    """
    src = src.float()
    tgt = tgt.float()

    if src.ndim != tgt.ndim:
        assert tgt.ndim == src.ndim + 1, (tgt.shape, src.shape)

        tgt, margin = _target_error_margin(tgt, mask, scale=margin_scale)
        margin /= 2

        src_log = src.clamp(eps).log()
        err_min = src_log - (tgt - margin).clamp(eps).log()
        err_max = src_log - (tgt + margin).clamp(eps).log()
        err = torch.min(err_min.abs(), err_max.abs())
    else:
        err = src.clamp(min=eps).log() - tgt.clamp(min=eps).log()

    # Compute the error
    err_mean, err_var = _mean_var_with_mask(err, mask, dim=dim, keepdim=True)
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
        margin_scale: float = _DEFAULT_MARGIN_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.dim = dim
        self.eps: float = eps
        self.margin_scale = margin_scale

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
        loss = compute_silog_loss(
            input, target, mask, self.dim, self.alpha, self.eps, self.margin_scale
        )
        return loss * self.scale


def compute_relative_loss(
    src: Tensor,
    tgt: Tensor,
    mask: Tensor,
    dim: T.Tuple[int, ...],
    eps: float = 1e-6,
    margin_scale: float = _DEFAULT_MARGIN_SCALE,
    square: bool = False,
) -> Tensor:
    """
    See :class:`RelativeLoss` for more details.

    This loss is a numerically stable version of the absolute relative error that
    can be used to supervise depth estimation models.
    """
    src = src.float()
    tgt = tgt.float()

    if src.ndim != tgt.ndim:
        assert tgt.ndim == src.ndim + 1, (tgt.shape, src.shape)
        tgt, margin = _target_error_margin(tgt, mask, scale=margin_scale)
        tgt[~mask] = 0.0
        margin[~mask] = 0.0
        margin /= 2

        err_min = (src - (tgt - margin).clamp_min(eps)).abs()
        err_max = (src - (tgt + margin).clamp_min(eps)).abs()
        err = torch.min(err_min, err_max)
    else:
        err = (src - tgt).abs()
    err = torch.where(tgt > 0, err / tgt, err)
    if square:
        err = err.square()
    err = _mean_with_mask(data=err.clamp_min(eps), mask=mask, dim=dim)
    if square:
        err = err.clamp(eps).sqrt()
    if err.ndim > 1:
        err = err.sum(dim=1)
    return err


class RelativeLoss(ScaledLossMixin, nn.Module):
    r"""
    Relative loss for depth estimation.
    """

    def __init__(
        self,
        dim: T.Tuple[int, ...] = (-2, -1),
        eps: float = 1e-6,
        margin_scale: float = _DEFAULT_MARGIN_SCALE,
        square: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps
        self.margin_scale = margin_scale
        self.square = square

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        See :func:`compute_absrel_loss` for more details.
        """
        err = compute_relative_loss(
            src, tgt, mask, self.dim, self.eps, self.margin_scale, self.square
        )
        return err * self.scale


def compute_mse_loss(
    src: Tensor,
    tgt: Tensor,
    mask: Tensor,
    dim: T.Tuple[int, ...],
    eps: float = 1e-6,
    margin_scale: float = _DEFAULT_MARGIN_SCALE,
):
    r"""
    Compute the Mean Squared Error (MSE) loss between the source and target
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
    margin_scale : float
        The scale of the margin of error. See :func:`_target_error_margin` for more details.
    """
    src = src.float()
    tgt = tgt.float()

    if src.ndim != tgt.ndim:
        assert tgt.ndim == src.ndim + 1, (tgt.shape, src.shape)
        tgt, margin = _target_error_margin(tgt, mask, scale=margin_scale)
        tgt[~mask] = 0.0
        margin[~mask] = 0.0
        margin /= 2

        err_min = (src - (tgt - margin)).abs()
        err_max = (src - (tgt + margin)).abs()
        err = torch.min(err_min, err_max)
    else:
        err = src - tgt

    err = _mean_with_mask(data=err.clamp_min(eps).square(), mask=mask, dim=dim)
    if err.ndim > 1:
        err = err.sum(dim=1)
    return err


class MSELoss(ScaledLossMixin, nn.Module):
    dim: T.Final[T.Tuple[int, ...]]
    eps: T.Final[float]

    def __init__(
        self,
        dim: T.Tuple[int, ...] = (-2, -1),
        eps: float = 1e-6,
        margin_scale: float = _DEFAULT_MARGIN_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps
        self.margin_scale = margin_scale

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
        # computing the MSE, and then scaling the loss such that the MSE is 1.0.
        scale = kwargs.pop("scale", 1.0)
        dim = kwargs.pop("dim", (-2, -1))
        n_points = 1000
        with torch.no_grad():
            tgt = torch.linspace(-1 / 2, 1 / 2, n_points, dtype=torch.float)
            src = tgt.flip(0) * init_scale

            dst_range = info.depth_max - info.depth_min
            dst_mean = dst_range / 2.0

            tgt = tgt * dst_range + dst_mean
            src = src * dst_range + dst_mean

            mse = cls(dim=(-1,), **kwargs)(src, tgt, torch.ones_like(src)).item()
            scale *= 1 / mse
        return cls(scale=scale, dim=dim, **kwargs)

    @TX.override
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        See :func:`compute_mse_loss` for more details.
        """
        err = compute_mse_loss(src, tgt, mask, self.dim, self.eps, self.margin_scale)
        return err * self.scale


class DCELoss(nn.Module):
    """
    Discretized Cross-Entropy (DCE) loss for Depth Estimation
    """

    def __init__(
        self,
        depth_normalize: T.Tuple[int, int],
        out_channel=200,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth_min = depth_normalize[0]
        self.depth_max = depth_normalize[1]
        self.bins_num = out_channel
        self.depth_min_log = torch.log10(torch.tensor(self.depth_min)).item()

        self.alpha = 2
        self.config_bins()
        self.noise_sample_ratio = 0.9
        self.eps = 1e-6

    def config_bins(self):
        # Modify some configs
        self.depth_bins_interval = (
            torch.log10(torch.tensor(self.depth_max)) - self.depth_min_log
        ) / self.bins_num
        bins_edges_in_log = (
            self.depth_min_log
            + self.depth_bins_interval
            * torch.tensor(
                list(range(self.bins_num))
                + [
                    self.bins_num,
                ]
            )
        )
        # bins_edges_in_log = torch.from_numpy(bins_edges_in_log)
        # The boundary of each bin
        # bins_edges_in_log = np.array([self.depth_min_log + self.depth_bins_interval * (i + 0.5)
        #                         for i in range(self.bins_num)])
        bins_weight = torch.tensor(
            [
                [np.exp(-self.alpha * (i - j) ** 2) for i in range(self.bins_num)]
                for j in np.arange(self.bins_num)
            ]
        ).cuda()
        self.register_buffer("bins_weight", bins_weight.float(), persistent=False)
        self.register_buffer(
            "bins_edges_in_log", bins_edges_in_log.float(), persistent=False
        )

    def depth_to_bins_in_log(self, depth, mask):
        """
        Discretize depth into depth bins. Predefined bins edges are in log space.
        Mark invalid padding area as bins_num + 1
        Args:
            @depth: 1-channel depth, [B, 1, h, w]
        return: depth bins [B, C, h, w]
        """
        invalid_mask = ~mask
        # depth[depth < self.depth_min] = self.depth_min
        # depth[depth > self.depth_max] = self.depth_max
        mask_lower = depth <= self.depth_min
        mask_higher = depth >= self.depth_max
        depth_bins_log = (
            (torch.log10(torch.abs(depth)) - self.depth_min_log)
            / self.depth_bins_interval
        ).to(torch.int)

        depth_bins_log[mask_lower] = 0
        depth_bins_log[mask_higher] = self.bins_num - 1
        depth_bins_log[depth_bins_log == self.bins_num] = self.bins_num - 1

        depth_bins_log[invalid_mask] = self.bins_num + 1
        return depth_bins_log

    def depth_to_bins(self, depth, mask, depth_edges, size_limite=(300, 300)):
        """
        Discretize depth into depth bins. Predefined bins edges are provided.
        Mark invalid padding area as bins_num + 1
        Args:
            @depth: 1-channel depth, [B, 1, h, w]
        return: depth bins [B, C, h, w]
        """

        def _depth_to_bins_block_(depth, mask, depth_edges):
            bins_id = torch.sum(
                depth_edges[:, None, None, None, :]
                < torch.abs(depth)[:, :, :, :, None],
                dim=-1,
            )
            bins_id = bins_id - 1
            invalid_mask = ~mask
            mask_lower = depth <= self.depth_min
            mask_higher = depth >= self.depth_max

            bins_id[mask_lower] = 0
            bins_id[mask_higher] = self.bins_num - 1
            bins_id[bins_id == self.bins_num] = self.bins_num - 1

            bins_id[invalid_mask] = self.bins_num + 1
            return bins_id

        _, _, H, W = depth.shape
        bins = mask.clone().long()
        h_blocks = np.ceil(H / size_limite[0]).astype(np.int)
        w_blocks = np.ceil(W / size_limite[1]).astype(np.int)
        for i in range(h_blocks):
            for j in range(w_blocks):
                h_start = i * size_limite[0]
                h_end_proposal = (i + 1) * size_limite[0]
                h_end = h_end_proposal if h_end_proposal < H else H
                w_start = j * size_limite[1]
                w_end_proposal = (j + 1) * size_limite[1]
                w_end = w_end_proposal if w_end_proposal < W else W
                bins_ij = _depth_to_bins_block_(
                    depth[:, :, h_start:h_end, w_start:w_end],
                    mask[:, :, h_start:h_end, w_start:w_end],
                    depth_edges,
                )
                bins[:, :, h_start:h_end, w_start:w_end] = bins_ij
        return bins

    @TX.override
    def forward(
        self, prediction, target, mask=None, pred_logit=None, **kwargs
    ):  # pred_logit, gt_bins, gt):
        B, _, H, W = target.shape
        if "bins_edges" not in kwargs or kwargs["bins_edges"] is None:
            # predefined depth bins in log space
            gt_bins = self.depth_to_bins_in_log(target, mask)
        else:
            bins_edges = kwargs["bins_edges"]
            gt_bins = self.depth_to_bins(target, mask, bins_edges)

        classes_range = torch.arange(
            self.bins_num, device=gt_bins.device, dtype=gt_bins.dtype
        )
        log_pred = torch.nn.functional.log_softmax(pred_logit, 1)
        log_pred = log_pred.reshape(B, log_pred.size(1), -1).permute((0, 2, 1))
        gt_reshape = gt_bins.reshape((B, -1))[:, :, None]
        one_hot = (gt_reshape == classes_range).to(
            dtype=torch.float, device=pred_logit.device
        )
        weight = torch.matmul(one_hot, self.bins_weight)
        weight_log_pred = weight * log_pred
        loss_pixeles = -torch.sum(weight_log_pred, dim=2)

        valid_pixels = torch.sum(mask).to(dtype=torch.float, device=pred_logit.device)
        loss = torch.sum(loss_pixeles) / (valid_pixels + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f"WCEL error, {loss}")
        return loss * self.loss_weight


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

        self.rmse = MSELoss()
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
