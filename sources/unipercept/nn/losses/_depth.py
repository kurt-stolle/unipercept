# from __future__ import annotations

from __future__ import annotations

import enum as E
import typing as T

import torch
import typing_extensions as TX
from torch import Tensor, nn

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
    alpha: float = 0.5,
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
        from unipercept.data.sets import Metadata, catalog

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
        return loss * self.scale


class HDSNRandomLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss.
    Replace the original grid masks with the random created masks.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))

    Paper: https://arxiv.org/pdf/2404.15506
    """

    def __init__(self, scale=1.0, random_num=32, sky_id=142, batch_limit=8, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.random_num = random_num
        self.sky_id = sky_id
        self.batch_limit = batch_limit
        self.eps = 1e-6

    def get_random_masks_for_batch(self, image_size: list) -> torch.Tensor:
        height, width = image_size
        crop_h_min = int(0.125 * height)
        crop_h_max = int(0.5 * height)
        crop_w_min = int(0.125 * width)
        crop_w_max = int(0.5 * width)
        h_max = height - crop_h_min
        w_max = width - crop_w_min
        crop_height = np.random.choice(
            np.arange(crop_h_min, crop_h_max), self.random_num, replace=False
        )
        crop_width = np.random.choice(
            np.arange(crop_w_min, crop_w_max), self.random_num, replace=False
        )
        crop_y = np.random.choice(h_max, self.random_num, replace=False)
        crop_x = np.random.choice(w_max, self.random_num, replace=False)
        crop_y_end = crop_height + crop_y
        crop_y_end[crop_y_end >= height] = height
        crop_x_end = crop_width + crop_x
        crop_x_end[crop_x_end >= width] = width

        mask_new = torch.zeros(
            (self.random_num, height, width), dtype=torch.bool, device="cuda"
        )  # .cuda() #[N, H, W]
        for i in range(self.random_num):
            mask_new[i, crop_y[i] : crop_y_end[i], crop_x[i] : crop_x_end[i]] = True

        return mask_new
        # return crop_y, crop_y_end, crop_x, crop_x_end

    def reorder_sem_masks(self, sem_label):
        # reorder the semantic mask of a batch
        assert sem_label.ndim == 3
        semantic_ids = torch.unique(
            sem_label[(sem_label > 0) & (sem_label != self.sky_id)]
        )
        sem_masks = [sem_label == id for id in semantic_ids]
        if len(sem_masks) == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out

        sem_masks = torch.cat(sem_masks, dim=0)
        mask_batch = torch.sum(sem_masks.reshape(sem_masks.shape[0], -1), dim=1) > 500
        sem_masks = sem_masks[mask_batch]
        if sem_masks.shape[0] > self.random_num:
            balance_samples = np.random.choice(
                sem_masks.shape[0], self.random_num, replace=False
            )
            sem_masks = sem_masks[balance_samples, ...]

        if sem_masks.shape[0] == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out

        if sem_masks.ndim == 2:
            sem_masks = sem_masks[None, :, :]
        return sem_masks

    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float("nan")
        target_nan[~mask_valid] = float("nan")

        valid_pixs = mask_valid.reshape((B, C, -1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = (
            target_nan.reshape((B, C, -1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1)
        )  # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median)).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = (
            prediction_nan.reshape((B, C, -1))
            .nanmedian(2, keepdims=True)[0]
            .unsqueeze(-1)
        )  # [b,c,h,w]
        pred_median[pred_median.isnan()] = 0
        pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans) * mask_valid)
        return loss_sum

    def conditional_ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        conditional_rank_ids = np.random.choice(B, B, replace=False)

        prediction_nan = prediction.clone()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float("nan")
        target_nan[~mask_valid] = float("nan")

        valid_pixs = mask_valid.reshape((B, C, -1)).sum(dim=2, keepdims=True) + self.eps
        valid_pixs = valid_pixs[:, :, :, None].contiguous()

        gt_median = (
            target_nan.reshape((B, C, -1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1)
        )  # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None].contiguous() / valid_pixs

        # in case some batches have no valid pixels
        gt_s_small_mask = gt_s < (torch.mean(gt_s) * 0.1)
        gt_s[gt_s_small_mask] = torch.mean(gt_s)
        gt_trans = (target - gt_median[conditional_rank_ids]) / (
            gt_s[conditional_rank_ids] + self.eps
        )

        pred_median = (
            prediction_nan.reshape((B, C, -1))
            .nanmedian(2, keepdims=True)[0]
            .unsqueeze(-1)
        )  # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape(
            (B, C, -1)
        )
        pred_s = pred_diff.sum(dim=2)[:, :, None, None].contiguous() / valid_pixs
        pred_s[gt_s_small_mask] = torch.mean(pred_s)
        pred_trans = (prediction - pred_median[conditional_rank_ids]) / (
            pred_s[conditional_rank_ids] + self.eps
        )

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans) * mask_valid)
        # print(torch.abs(gt_trans - pred_trans)[mask_valid])
        return loss_sum

    def forward(self, prediction, target, mask=None, sem_mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape

        loss = 0.0
        valid_pix = 0.0

        device = target.device

        batches_dataset = kwargs["dataset"]
        self.batch_valid = torch.tensor(
            [
                1 if batch_dataset not in self.disable_dataset else 0
                for batch_dataset in batches_dataset
            ],
            device=device,
        )[:, None, None, None]

        batch_limit = self.batch_limit

        random_sample_masks = self.get_random_masks_for_batch((H, W))  # [N, H, W]
        for i in range(B):
            # each batch
            mask_i = mask[i, ...]  # [1, H, W]
            if self.batch_valid[i, ...] < 0.5:
                loss += 0 * torch.sum(prediction[i, ...])
                valid_pix += 0 * torch.sum(mask_i)
                continue

            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)

            # get semantic masks
            sem_label_i = sem_mask[i, ...] if sem_mask is not None else None
            if sem_label_i is not None:
                sem_masks = self.reorder_sem_masks(sem_label_i)  # [N, H, W]
                random_sem_masks = torch.cat([random_sample_masks, sem_masks], dim=0)
            else:
                random_sem_masks = random_sample_masks
            # random_sem_masks = random_sample_masks

            sampled_masks_num = random_sem_masks.shape[0]
            loops = int(np.ceil(sampled_masks_num / batch_limit))
            conditional_rank_ids = np.random.choice(
                sampled_masks_num, sampled_masks_num, replace=False
            )

            for j in range(loops):
                mask_random_sem_loopi = random_sem_masks[
                    j * batch_limit : (j + 1) * batch_limit, ...
                ]
                mask_sample = (mask_i & mask_random_sem_loopi).unsqueeze(
                    1
                )  # [N, 1, H, W]
                loss += self.ssi_mae(
                    prediction=pred_i[: mask_sample.shape[0], ...],
                    target=target_i[: mask_sample.shape[0], ...],
                    mask_valid=mask_sample,
                )
                valid_pix += torch.sum(mask_sample)

                # conditional ssi loss
                # rerank_mask_random_sem_loopi = random_sem_masks[conditional_rank_ids, ...][j*batch_limit:(j+1)*batch_limit, ...]
                # rerank_mask_sample = (mask_i & rerank_mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                # loss_cond = self.conditional_ssi_mae(
                #     prediction=pred_i[:rerank_mask_sample.shape[0], ...],
                #     target=target_i[:rerank_mask_sample.shape[0], ...],
                #     mask_valid=rerank_mask_sample)
                # print(loss_cond / (torch.sum(rerank_mask_sample) + 1e-10), loss_cond, torch.sum(rerank_mask_sample))
                # loss += loss_cond
                # valid_pix += torch.sum(rerank_mask_sample)

        # crop_y, crop_y_end, crop_x, crop_x_end = self.get_random_masks_for_batch((H, W)) # [N,]
        # for j in range(B):
        #     for i in range(self.random_num):
        #         mask_crop = mask[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...] #[1, 1, crop_h, crop_w]
        #         target_crop = target[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...]
        #         pred_crop = prediction[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...]
        #         loss += self.ssi_mae(prediction=pred_crop, target=target_crop, mask_valid=mask_crop)
        #         valid_pix += torch.sum(mask_crop)

        # the whole image
        mask = mask * self.batch_valid.bool()
        loss += self.ssi_mae(prediction=prediction, target=target, mask_valid=mask)
        valid_pix += torch.sum(mask)
        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f"HDSNL NAN error, {loss}, valid pix: {valid_pix}")
        return loss * self.scale


class HDNRandomLoss(nn.Module):
    """
    Hieratical depth normalization loss. Replace the original hieratical depth ranges with randomly sampled ranges.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))

    Paper: https://arxiv.org/pdf/2404.15506
    """

    def __init__(self, scale=1, random_num=32, **kwargs):
        super(HDNRandomLoss, self).__init__()
        self.scale = scale
        self.random_num = random_num
        self.eps = 1e-6

    def get_random_masks_for_batch(
        self, depth_gt: torch.Tensor, mask_valid: torch.Tensor
    ) -> torch.Tensor:
        valid_values = depth_gt[mask_valid]
        max_d = valid_values.max().item() if valid_values.numel() > 0 else 0.0
        min_d = valid_values.min().item() if valid_values.numel() > 0 else 0.0

        sample_min_d = (
            np.random.uniform(0, 0.75, self.random_num) * (max_d - min_d) + min_d
        )
        sample_max_d = (
            np.random.uniform(sample_min_d + 0.1, 1 - self.eps, self.random_num)
            * (max_d - min_d)
            + min_d
        )

        mask_new = [
            (depth_gt >= sample_min_d[i])
            & (depth_gt < sample_max_d[i] + 1e-30)
            & mask_valid
            for i in range(self.random_num)
        ]
        mask_new = torch.stack(mask_new, dim=0).cuda()  # [N, 1, H, W]
        return mask_new

    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float("nan")
        target_nan[~mask_valid] = float("nan")

        valid_pixs = mask_valid.reshape((B, C, -1)).sum(dim=2, keepdims=True) + self.eps
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = (
            target_nan.reshape((B, C, -1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1)
        )  # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = (
            prediction_nan.reshape((B, C, -1))
            .nanmedian(2, keepdims=True)[0]
            .unsqueeze(-1)
        )  # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape(
            (B, C, -1)
        )
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans) * mask_valid)
        return loss_sum

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape

        loss = 0.0
        valid_pix = 0.0

        device = target.device

        batches_dataset = kwargs["dataset"]
        self.batch_valid = torch.tensor(
            [
                1 if batch_dataset not in self.disable_dataset else 0
                for batch_dataset in batches_dataset
            ],
            device=device,
        )[:, None, None, None]

        batch_limit = 4
        loops = int(np.ceil(self.random_num / batch_limit))
        for i in range(B):
            mask_i = mask[i, ...]  # [1, H, W]

            if self.batch_valid[i, ...] < 0.5:
                loss += 0 * torch.sum(prediction[i, ...])
                valid_pix += 0 * torch.sum(mask_i)
                continue

            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            mask_random_drange = self.get_random_masks_for_batch(
                target[i, ...], mask_i
            )  # [N, 1, H, W]
            for j in range(loops):
                mask_random_loopi = mask_random_drange[
                    j * batch_limit : (j + 1) * batch_limit, ...
                ]
                loss += self.ssi_mae(
                    prediction=pred_i[: mask_random_loopi.shape[0], ...],
                    target=target_i[: mask_random_loopi.shape[0], ...],
                    mask_valid=mask_random_loopi,
                )
                valid_pix += torch.sum(mask_random_loopi)

        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f"HDNL NAN error, {loss}, valid pix: {valid_pix}")
        return loss * self.scale


class VNLoss(ScaledLossMixin, nn.Module):
    """
    Virtual Normal Loss.

    Paper: https://arxiv.org/pdf/2103.04216
    """

    def __init__(
        self,
        delta_cos=0.867,
        delta_diff_x=0.01,
        delta_diff_y=0.01,
        delta_diff_z=0.01,
        delta_z=1e-5,
        sample_ratio=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.eps = 1e-6

    def init_image_coor(self, intrinsic, height, width):
        # x_row = torch.arange(0, W, device="cuda")
        # x = torch.tile(x_row, (H, 1))
        # x = x.to(torch.float32)
        # u_m_u0 = x[None, None, :, :] - u0
        # self.register_buffer('u_m_u0', u_m_u0, persistent=False)

        # y_col = torch.arange(0, H, device="cuda")  # y_col = np.arange(0, height)
        # y = torch.transpose(torch.tile(y_col, (W, 1)), 1, 0)
        # y = y.to(torch.float32)
        # v_m_v0 = y[None, None, :, :] - v0
        # self.register_buffer('v_m_v0', v_m_v0, persistent=False)

        # pix_idx_mat = torch.arange(H*W, device="cuda").reshape((H, W))
        # self.register_buffer('pix_idx_mat', pix_idx_mat, persistent=False)
        # self.pix_idx_mat = torch.arange(height*width, device="cuda").reshape((height, width))

        u0 = intrinsic[:, 0, 2][:, None, None, None]
        v0 = intrinsic[:, 1, 2][:, None, None, None]
        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device="cuda"),
                torch.arange(0, width, dtype=torch.float32, device="cuda"),
            ],
            indexing="ij",
        )
        u_m_u0 = x[None, None, :, :] - u0
        v_m_v0 = y[None, None, :, :] - v0
        # return u_m_u0, v_m_v0
        self.register_buffer("v_m_v0", v_m_v0, persistent=False)
        self.register_buffer("u_m_u0", u_m_u0, persistent=False)

    def transfer_xyz(self, depth, focal_length, u_m_u0, v_m_v0):
        x = u_m_u0 * depth / focal_length
        y = v_m_v0 * depth / focal_length
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]
        return pw

    def select_index(self, B, H, W, mask):
        """ """
        p1 = []
        p2 = []
        p3 = []
        pix_idx_mat = torch.arange(H * W, device="cuda").reshape((H, W))
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, mask[i, ...].gt(self.eps))
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(H * W * self.sample_ratio)
            sample_num = (
                intend_sample_num
                if num_effect_pixels >= intend_sample_num
                else num_effect_pixels
            )

            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p2i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p3i = inputs_index[shuffle_effect_pixels[:sample_num]]

            cat_null = torch.tensor(
                (
                    [
                        0,
                    ]
                    * (intend_sample_num - sample_num)
                ),
                dtype=torch.long,
                device="cuda",
            )
            p1i = torch.cat([p1i, cat_null])
            p2i = torch.cat([p2i, cat_null])
            p3i = torch.cat([p3i, cat_null])

            p1.append(p1i)
            p2.append(p2i)
            p3.append(p3i)

        p1 = torch.stack(p1, dim=0)
        p2 = torch.stack(p2, dim=0)
        p3 = torch.stack(p3, dim=0)

        p1_x = p1 % W
        p1_y = torch.div(p1, W, rounding_mode="trunc").long()  # p1 // W

        p2_x = p2 % W
        p2_y = torch.div(p2, W, rounding_mode="trunc").long()  # p2 // W

        p3_x = p3 % W
        p3_y = torch.div(p3, W, rounding_mode="trunc").long()  # p3 // W
        p123 = {
            "p1_x": p1_x,
            "p1_y": p1_y,
            "p2_x": p2_x,
            "p2_y": p2_y,
            "p3_x": p3_x,
            "p3_y": p3_y,
        }
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        B, _, _, _ = pw.shape
        p1_x = p123["p1_x"]
        p1_y = p123["p1_y"]
        p2_x = p123["p2_x"]
        p2_y = p123["p2_y"]
        p3_x = p123["p3_x"]
        p3_y = p123["p3_y"]

        pw_groups = []
        for i in range(B):
            pw1 = pw[i, p1_y[i], p1_x[i], :]
            pw2 = pw[i, p2_y[i], p2_x[i], :]
            pw3 = pw[i, p3_y[i], p3_x[i], :]
            pw_bi = torch.stack([pw1, pw2, pw3], dim=2)
            pw_groups.append(pw_bi)
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.stack(pw_groups, dim=0)
        return pw_groups

    def filter_mask(
        self,
        p123,
        gt_xyz,
        delta_cos=0.867,
        delta_diff_x=0.005,
        delta_diff_y=0.005,
        delta_diff_z=0.005,
    ):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat(
            [
                pw12[:, :, :, None],
                pw13[:, :, :, None],
                pw23[:, :, :, None],
            ],
            3,
        )  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = (
            pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1).contiguous()
        )  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.contiguous().view(
            m_batchsize * groups, -1, index
        )  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(
            q_norm.contiguous().view(m_batchsize * groups, index, 1),
            q_norm.view(m_batchsize * groups, 1, index),
        )  # []
        energy = torch.bmm(
            proj_query, proj_key
        )  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + self.eps)
        norm_energy = norm_energy.contiguous().view(m_batchsize * groups, -1)
        mask_cos = (
            torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        )  # igonre
        mask_cos = mask_cos.contiguous().view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth, intrinsic, mask):
        B, C, H, W = gt_depth.shape
        focal_length = intrinsic[:, 0, 0][:, None, None, None]
        u_m_u0, v_m_v0 = (
            self.u_m_u0,
            self.v_m_v0,
        )  # self.init_image_coor(intrinsic, height=H, width=W)

        pw_gt = self.transfer_xyz(gt_depth, focal_length, u_m_u0, v_m_v0)
        pw_pred = self.transfer_xyz(pred_depth, focal_length, u_m_u0, v_m_v0)

        p123 = self.select_index(B, H, W, mask)
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(
            p123,
            pw_gt,
            delta_cos=0.867,
            delta_diff_x=0.005,
            delta_diff_y=0.005,
            delta_diff_z=0.005,
        )

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = (
            mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2).contiguous()
        )
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    @TX.override
    def forward(self, prediction, target, mask, intrinsic, select=True, **kwargs):
        # configs for the cameras
        # focal_length = intrinsic[:, 0, 0][:, None, None, None]
        # u0 = intrinsic[:, 0, 2][:, None, None, None]
        # v0 = intrinsic[:, 1, 2][:, None, None, None]
        B, _, H, W = target.shape
        if (
            "u_m_u0" not in self._buffers
            or "v_m_v0" not in self._buffers
            or self.u_m_u0.shape != torch.Size([B, 1, H, W])
            or self.v_m_v0.shape != torch.Size([B, 1, H, W])
        ):
            self.init_image_coor(intrinsic, H, W)

        gt_points, pred_points = self.select_points_groups(
            target, prediction, intrinsic, mask
        )

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        pred_p12 = pred_points[:, :, :, 1] - pred_points[:, :, :, 0]
        pred_p13 = pred_points[:, :, :, 2] - pred_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        pred_normal = torch.cross(pred_p12, pred_p13, dim=2)
        pred_norm = torch.norm(pred_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        pred_mask = pred_norm == 0.0
        gt_mask = gt_norm == 0.0
        pred_mask = pred_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        pred_mask *= self.eps
        gt_mask *= self.eps
        gt_norm = gt_norm + gt_mask
        pred_norm = pred_norm + pred_mask
        gt_normal = gt_normal / gt_norm
        pred_normal = pred_normal / pred_norm
        loss = torch.abs(gt_normal - pred_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25) :]
        loss = torch.sum(loss) / (loss.numel() + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f"VNL NAN error, {loss}")
        return loss * self.scale


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
