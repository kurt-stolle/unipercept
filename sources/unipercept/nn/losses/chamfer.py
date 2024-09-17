from __future__ import annotations

import torch
import typing_extensions as TX
from torch import nn

from unipercept.nn.losses.mixins import ScaledLossMixin
from unipercept.types import Tensor
from unipercept.vision.knn_points import knn_gather, knn_points


def _handle_pointcloud_input(
    points: Tensor, lengths: Tensor | None, normals: Tensor | None
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if points.ndim != 3:
        msg = f"Expected points to be of shape (N, P, D). Got: {points.shape}."
        raise ValueError(msg)
    X = points
    if lengths is not None:
        if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
            msg = f"Expected lengths to be of shape (N,). Got: {lengths.shape}."
            raise ValueError(msg)
        if lengths.max() > X.shape[1]:
            raise ValueError("A length value was too long")
    if lengths is None:
        lengths = torch.full(
            (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
        )
    if normals is not None and normals.ndim != 3:
        msg = f"Expected normals to be of shape (N, P, 3). Got: {normals.shape}."
        raise ValueError(msg)
    return X, lengths, normals


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights,
    batch_reduction: str | None,
    point_reduction: str | None,
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = nn.functional.cosine_similarity(
            x_normals, x_normals_near, dim=2, eps=1e-6
        )
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if batch_reduction == "mean":
                div = weights.sum() if weights is not None else max(N, 1)
                cham_x /= div
                if return_normals:
                    cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals


def compute_chamfer_loss(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: str | None = "mean",
    point_reduction: str | None = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Parameters
    ----------
    x:
        FloatTensor of shape (N, P1, D) or a Pointclouds object representing
        a batch of point clouds with at most P1 points in each batch element,
        batch size N and feature dimension D.
    y:
        FloatTensor of shape (N, P2, D) or a Pointclouds object representing
        a batch of point clouds with at most P2 points in each batch element,
        batch size N and feature dimension D.
    x_lengths:
        Optional LongTensor of shape (N,) giving the number of points in each
        cloud in x.
    y_lengths:
        Optional LongTensor of shape (N,) giving the number of points in each
        cloud in y.
    x_normals:
        Optional FloatTensor of shape (N, P1, D).
    y_normals:
        Optional FloatTensor of shape (N, P2, D).
    weights:
        Optional FloatTensor of shape (N,) giving weights for
        batch elements for reduction operation.
    batch_reduction:
        Reduction operation to apply for the loss across the
        batch, can be one of ["mean", "sum"] or None.
    point_reduction:
        Reduction operation to apply for the loss across the
        points, can be one of ["mean", "sum"] or None.
    norm:
        int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
    single_directional:
        If False (default), loss comes from both the distance between
        each point in x and its nearest neighbor in y and each point in y and its nearest
        neighbor in x. If True, loss is the distance between each point in x and its
        nearest neighbor in y.
    abs_cosine:
        If False, loss_normals is from one minus the cosine similarity.
        If True (default), loss_normals is from one minus the absolute value of the
        cosine similarity, which means that exactly opposite normals are considered
        equivalent to exactly matching normals, i.e. sign does not matter.

    Returns
    -------
    Tensor
        Reduced distance (loss) between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    Tensor
        Reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    assert norm in (1, 2), f"{norm=} not in {{1, 2}}"
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    cham_y, cham_norm_y = _chamfer_distance_single_direction(
        y,
        x,
        y_lengths,
        x_lengths,
        y_normals,
        x_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if point_reduction is not None:
        return (
            cham_x + cham_y,
            (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
        )
    return (
        (cham_x, cham_y),
        (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
    )


class ChamferLoss(ScaledLossMixin, nn.Module):
    def __init__(
        self,
        batch_reduction: str | None = "mean",
        point_reduction: str | None = "mean",
        norm: int = 2,
        single_directional: bool = False,
        abs_cosine: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_reduction = batch_reduction
        self.point_reduction = point_reduction
        self.norm = norm
        self.single_directional = single_directional
        self.abs_cosine = abs_cosine

    @TX.override
    def forward(
        self,
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
    ):
        return compute_chamfer_loss(
            x,
            y,
            x_lengths,
            y_lengths,
            x_normals,
            y_normals,
            weights,
            self.batch_reduction,
            self.point_reduction,
            self.norm,
            self.single_directional,
            self.abs_cosine,
        )
