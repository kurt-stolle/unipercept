"""
Implements point sampling and selection methods proposed in PointRend [1]_.

See Also
--------
- `Reference implementation <https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend>`_ provided in [1]_.

References
----------
[1] `PointRend: Image Segmentation as Rendering <https://arxiv.org/abs/1912.08193>`_
"""

from __future__ import annotations

import enum as E
import math
import typing as T

import torch
import torch.fx
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn

from unipercept.utils.check import assert_shape, assert_tensor
from unipercept.vision.coord import generate_coord_grid
from unipercept.vision.filter import filter2d


def distance_transform(
    input_map: torch.Tensor, kernel_size: int = 3, h: float = 0.35
) -> torch.Tensor:
    r"""
    Very simple distance transform using convolutions. This is a naive implementation
    and may be removed in the future in favor of a more efficient implementation.
    """
    assert_tensor(input_map)
    assert_shape(input_map, (..., "C", "H", "W"))
    assert kernel_size % 2 == 1 and kernel_size > 0, kernel_size

    ndim = input_map.ndim
    if ndim == 3:
        input_map = input_map.unsqueeze(1)  # add channel

    *BATCH, C, H, W = input_map.shape
    input_map = input_map.view(-1, C, H, W)

    if not input_map.is_floating_point():
        input_map = input_map.float()
    grid = generate_coord_grid(
        (kernel_size, kernel_size),
        device=input_map.device,
        dtype=input_map.dtype,
    ).unsqueeze(0)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    dist_map = torch.zeros_like(input_map)
    boundary = input_map.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(math.ceil(max(H, W) / math.floor(kernel_size / 2))):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        dist_map += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    dist_map = dist_map.view(*BATCH, C, H, W)
    if ndim == 3:
        dist_map = dist_map.squeeze(1)
    return dist_map


@torch.no_grad()
def sparse_fill(input_map: Tensor, valid_mask: Tensor | None = None, d_coord=2):
    if valid_mask is None:
        valid_mask = ~torch.isnan(input_map)

    spatial_size = input_map.shape[-d_coord:]
    input_flat = input_map.view(-1, *spatial_size)
    filled_flat = input_map.clone().reshape(-1, *spatial_size)
    valid_flat = valid_mask.detach().cpu().reshape(-1, *spatial_size)

    for i in range(input_flat.size(0)):
        # Get the coordinates of the nearest non-negative value for each pixel
        nearest_coords = distance_transform_edt(
            ~(valid_flat[i]), return_distances=False, return_indices=True
        )
        nearest_coords = torch.from_numpy(nearest_coords).to(input_map.device)
        filled_flat[i] = input_flat[i][tuple(nearest_coords)]

    return filled_flat.view_as(input_map)


def sample(input, point_coords, align_corners: bool = False, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Parameters
    ----------
    input: Tensor
        A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
    point_coords: Tensor
        A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns
    -------
    output: Tensor
        A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
        features for points in `point_coords`.
        The features are obtained via interpolation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    assert_shape(input, ("B", "C", "H", "W"))
    assert_shape(point_coords, (..., 2))

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = nn.functional.grid_sample(
        input,
        2.0 * point_coords - 1.0,
        align_corners=align_corners,
        padding_mode="border",
        **kwargs,
    )
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def logit_uncertainty(logits: Tensor) -> Tensor:
    """
    Method proposed in PointRend [1]_ to calculates the uncertainty of prediction logits.

    See :function:`random_points_with_importance`, where this is the default importance
    function.
    """
    return -logits.detach().abs()


def random_points(
    source: Tensor,
    n_points: int,
    d_coord: int = 2,
    mask: Tensor | None = None,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space.

    The points are sampled uniformly can can optionally be conditioned on a mask that
    defines the region of interest.

    Parameters
    ----------
    source: Tensor
        A tensor of shape $(N, C, ...)$
    n_points: int
        The number of points $P$ to sample.
    d_coord: int
        The number of dimensions $S$ of the coordinate space, by default 2d.
    mask: Tensor, optional
        A boolean tensor of shape $(N, 1, ...)$ that contains a mask of the region of
        interest. Random points will only be selected such that they lie inside the mask.

    Returns
    -------
    Tensor
        A tensor of shape $(N, P, S)$ that contains $S$d coordinates of $P$ points.
    """
    assert_shape(source, ("B", "C", *(f"C_{i}" for i in range(d_coord))))

    n_batch = source.size(0)

    # Sample random coordinates in [0, 1] for `d_coord` spatial dimensions.
    point_coords = torch.rand(
        n_batch, n_points, d_coord, dtype=torch.float32, device=source.device
    )

    # No mask constaints, return the coordinates directly.
    if mask is None:
        return point_coords

    # Currently only 2D masks are supported
    if d_coord != 2:
        msg = "Mask constraints are only supported for 2D coordinates."
        raise NotImplementedError(msg)

    # Ensure the sampled points lie inside the mask by sampling random pixel coordinates
    # from the mask and adding the point coordinates normalized by the spatial size
    # as additive noise
    mask_flat = mask.view(n_batch, -1)
    mask_cumsum = mask_flat.cumsum(dim=1)

    # Generate uniform samples from 0 to the max of the cumsum
    max_cumsum = mask_cumsum[:, -1].unsqueeze(1)
    random_samples = torch.rand(n_batch, n_points, device=source.device) * max_cumsum

    # Find the indices in the cumulative sum that correspond to each random sample
    idx = torch.searchsorted(mask_cumsum, random_samples)

    # Convert linear indices to subscripts
    idx_subscripts = torch.stack(
        (
            torch.div(idx, mask.shape[-1], rounding_mode="trunc"),
            idx % mask.shape[-1],
        ),
        dim=-1,
    )

    # Convert indices to pixel (center) coordinates
    idx_subscripts = idx_subscripts.float() + 0.5
    point_coords = point_coords - 0.5

    # Convert the pixel indices to normalized coordinates
    spatial_size = torch.tensor(
        mask.shape[-d_coord:], dtype=torch.float32, device=source.device
    )

    # Flip X and Y coordinates
    idx_subscripts = idx_subscripts.flip(-1)
    spatial_size = spatial_size.flip(-1)

    # Normalize and add uniform noise within each pixel
    pixel_coords = idx_subscripts.float() / spatial_size
    pixel_coords = pixel_coords
    point_noise = point_coords / spatial_size

    return pixel_coords + point_noise


def random_points_with_importance(
    source: Tensor,
    n_points: int,
    d_coord: int = 2,
    mask: Tensor | None = None,
    oversample_ratio: float = 3,
    importance_fn: T.Callable[[Tensor], Tensor] = logit_uncertainty,
    importance_sample_ratio: float = 0.75,
    **kwargs,
):
    """
    Sample points in $[0, 1]$ x [0, 1]$ coordinate space based on their importance, which
    PointRend [1]_ uses to supervise imports based on their uncertainty.

    Notes
    -----

    It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    Calculating uncertainties of the coarse predictions first and sampling them for points leads
    to incorrect results.

    To illustrate this: assume importance_fn(src)=-abs(src), a sampled point between
    two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    However, if we calculate uncertainties for the coarse predictions first,
    both will have -1 uncertainty, and the sampled point will get -1 uncertainty.

    Parameters
    ----------
    source: Tensor
        A tensor of shape $(N, C, ...)$
    n_points: int
        The number of points $P$ to sample.
    d_coord: int
        The number of dimensions $S$ of the coordinate space, by default 2d.
    mask: Tensor, optional
        A boolean tensor of shape $(N, 1, ...)$ that contains a mask of the region of
        interest. Random points will only be selected such that they lie inside the mask.
    oversample_ratio: int
        Oversampling parameter, controls the amount of points $P_s$ over which
        the importance is computed.
    importance_sample_ratio: float
        Ratio of points that are sampled via importance sampling.
    importance_fn: Callable[[Tensor], Tensor]
        A function that takes samples points of shape $(N, C, P_s)$ that returns their
        importance as $(N, 1, P_s)$.
    **kwargs:
        Additional arguments for :function:`torch.nn.functional.grid_sample`.

    Returns
    -------
    Tensor
        A tensor of shape $(N, P, S)$ that contains $S$d coordinates of $P$ points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

    n_batch = source.shape[0]
    n_sample = int(n_points * oversample_ratio)

    point_coords = random_points(source, n_sample, d_coord, mask)
    samples = sample(source, point_coords, **kwargs)
    point_importance = importance_fn(samples)

    # Select the most important points
    n_important = int(importance_sample_ratio * n_points)
    i_important = torch.topk(point_importance[:, 0, :], k=n_important, dim=1)[1]

    shift = n_sample * torch.arange(n_batch, dtype=torch.long, device=source.device)
    i_important += shift[:, None]

    point_coords = point_coords.view(-1, 2)[i_important.view(-1), :].view(
        n_batch, n_important, d_coord
    )

    # Fill the rest with random points to ensure a total of `n_points`
    n_random = n_points - n_important
    if n_random > 0:
        point_coords = torch.cat(
            (point_coords, random_points(source, n_random, d_coord, mask)),
            dim=1,
        )

    # The result is of shape (n_batch, n_points, 2)
    return point_coords


class BinsMode(E.StrEnum):
    """
    An enumeration that defines the mode of binning.
    """

    QUANTILE = E.auto()
    LINEAR = E.auto()


@torch.no_grad()
def bins_by_values(
    source: Tensor,
    n_bins: int,
    mask: Tensor | None = None,
    use_batch: bool = False,
    mode: BinsMode | T.Literal["linear", "quantile"] = BinsMode.QUANTILE,
    eps: float = 1e-6,
) -> Tensor:
    r"""
    Create linear bins based on quantiles of the source tensor.

    Parameters
    ----------
    source: Tensor[B, C, *]
        A tensor of shape $(B, ...)$
    n_bins: int
        The number of bins $N$ to create.
    mask: Tensor[B, *], optional
        A mask of shape $(B, ...)$ that defines the valid region of the source tensor.
    use_batch: bool
        If set to `True`, the bins are computed based on the per-batch values. Otherwise
        the bins are computed based on the global values.
    """

    assert_shape(source, ("B", "C", ...))

    # Batch dims
    n_batch = source.shape[0]

    # Compute global or per-batch bins?
    if use_batch:
        source = source.flatten(1)

        return torch.cat(
            [
                bins_by_values(
                    source[i].unsqueeze(0),
                    n_bins,
                    mask[i].unsqueeze(0) if mask is not None else None,
                    use_batch=False,
                    mode=mode,
                )
                for i in range(n_batch)
            ],
            dim=0,
        )

    # Filter out the masked values
    if mask is not None:
        source = source[mask]
    else:
        source = source[~source.isnan()]

    # Compute the bins based on the selected mode
    match mode:
        case BinsMode.QUANTILE:
            quantiles = torch.linspace(0, 1, n_bins + 1, device=source.device)[1:]
            bins = torch.quantile(source, quantiles) + eps
        case BinsMode.LINEAR:
            bins = torch.linspace(source.min(), source.max() + eps, n_bins + 1)[1:]
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    bins = bins.unsqueeze(0).repeat(n_batch, 1)

    return bins


def map_in_bins(
    source: Tensor,
    bins: Tensor,
    fn: T.Callable[[Tensor, Tensor], Tensor],
    mask: Tensor | None = None,
    values: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """
    Apply a function to a source tensor by dividing it into ``n_bins`` bins and applying
    the function ``fn`` to each bin.

    The resulting tensor is concatenated along the channel (C) dimension.

    Parameters
    ----------
    source: Tensor[B, C, *]
        A tensor whose values are divided into bins.
    bins: Tensor[B, N]
        A tensor that contains the bin boundaries.
    fn: Callable[[Tensor[B, C, *], Tensor[B, C, *]], Tensor[B, R, *]]
        A function that is applied to each bin of the source tensor. Returns :math:`R`
        channels, which are concatenated into :math:`C \times R` output channels.
        The first argument is the source tensor, and the second argument is a boolean
        mask of the same shape that indicates which values belong to the current bin.
    mask: Tensor[B, *], optional
        Mask for invalid values, which are combined with the binned mask usign a logcal AND.
    values: Tensor[B, *], optional
        A tensor that contains the values to be binned. If not provided, the source tensor
        is used.

    Returns
    -------
    Tensor[B, N * R, *]
        Resulting tensor after applying the function to each bin.
    """

    assert_shape(source, ("B", "C", ...))
    assert_shape(bins, ("B", "N"))

    n_bins = bins.shape[1]

    # Create a mask that indicates which bin a value belongs to
    if values is None:
        values = source
    mask_bin = torch.searchsorted(bins, values, right=True)

    # Apply the function to each bin
    outputs = []
    for i in range(n_bins):
        mask_i = mask_bin == i
        if mask is not None:
            mask_i = mask_i & mask
        if not mask_i.any():
            continue
        output_i = fn(source, mask_i)
        outputs.append(output_i)

    # Concatenate the results along the channel dimension
    return torch.cat(outputs, dim=1)


def random_points_with_bins(
    source: Tensor,
    n_points: int,
    d_coord: int = 2,
    mask: Tensor | None = None,
    n_bins: int = 10,
    use_batch: bool = False,
    mode: BinsMode | T.Literal["linear", "quantile"] = BinsMode.QUANTILE,
    values: Tensor | None = None,
):
    """
    Samples points uniformly from the source tensor, where the points are first divided
    into bins based on their values.

    See :function:`random_points` for more details.
    """

    if values is None:
        values = source
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)

    bins = bins_by_values(source, n_bins, mask=mask, mode=mode, use_batch=use_batch)

    points = []
    for s, m, b, v in zip(source, mask, bins, values, strict=True):
        points_per_bin = math.ceil(
            n_points
            / map_in_bins(s, b, lambda _, v: v.any(), mask=m, values=v)
            .int()
            .sum()
            .item()
        )

        points.append(
            map_in_bins(
                s,
                b,
                lambda s, m: random_points(s, points_per_bin, d_coord, mask=m),
                mask=m,
            )[:, :n_points, ...]
        )
    return torch.stack(points, dim=0)
