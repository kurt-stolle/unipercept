"""
Implements point sampling and selection methods proposed in PointRend [1]_.

See Also
--------
- `Reference implementation <https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend>`_ provided in [1]_.

References
----------
[1] `PointRend: Image Segmentation as Rendering <https://arxiv.org/abs/1912.08193>`_
"""

import typing as T

import torch
from torch import nn, Tensor


def point_sample(input, point_coords, align_corners: bool = False, **kwargs):
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


def logit_uncertainty(logits: Tensor) -> Tensor:
    """
    Method proposed in PointRend [1]_ to calculates the uncertainty of prediction logits.

    See :function:`random_points_with_importance`, where this is the default importance
    function.
    """
    return -logits.abs()


def random_points(
    source: Tensor, n_points: int, d_coord: int = 2, mask: Tensor | None = None
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
    assert source.ndim == 2 + d_coord, (source.shape, d_coord)
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
    point_samples = point_sample(source, point_coords, **kwargs)
    point_importance = importance_fn(point_samples)

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
