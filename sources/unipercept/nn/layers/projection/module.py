from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from . import render


class DepthProjection(nn.Module):
    """Maps the world-positon of an object using a projection of the object 2D mask (i.e. segmentation) and estimated mean  depth."""

    max_depth: Tensor

    def __init__(
        self,
        max_depth: Tensor | float,
        scale: float = 4.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_depth
            Maximum depth for denoramlization of the mean depth, which should be
            fixed for a given dataset.
        scale
            Value to upscale screen by, e.g. when the network gives masks
            with size (h,w) then they will be projected as if they had size
            (h*scale, w*scale). This is useful because predictions are often
            made on a downscaled version of the image.
        """

        super().__init__(**kwargs)

        self.scale = scale
        self.threshold = 0.2

        self.register_buffer(
            "max_depth",
            torch.tensor(max_depth, dtype=torch.float32, requires_grad=False),
            persistent=False,
        )

    @override
    def forward(
        self, logits: Tensor, mean_depths: Tensor, cams: render.Cameras
    ) -> Tensor:
        """ "
        Project the masks and mean depths to world coordinates.

        Parameters
        ----------
        masks
            Instance masks of shape (n, h, w).
        mean_depths
            Mean depth of each instance of shape (n,).
        K
            Camera intrinsics of shape (3, 3).

        Returns
        -------
            Points in world coordinates of shape (n, 3).
        """
        masks = logits.sigmoid() > self.threshold
        if len(masks) == 0:
            return torch.empty((0, 3), dtype=self.max_depth.dtype, device=masks.device)

        points_xyd = self._read_points_xyd(masks, mean_depths)
        points_world = self._unproject(cams, points_xyd)

        return points_world

    def _read_points_xyd(self, masks: Tensor, mean_depths: Tensor) -> Tensor:
        """
        Create a list of points using the mass center and mean depth of each
        instance.
        """

        assert masks.ndim == 3, masks.ndim  # n, h, w
        assert mean_depths.ndim == 2, mean_depths.ndim  # n x d
        assert len(masks) == len(mean_depths), (len(masks), len(mean_depths))

        indices, yx = masks.argwhere().split([1, 2], dim=1)
        _, counts = torch.unique(indices, return_counts=True)
        indices_len: list[int] = counts.tolist()

        # Convert yx -> xy
        xy = torch.flip(yx, [1])

        # Compensate for downscaling of masks
        xy = xy * self.scale

        # Split indices
        indices_split = indices.float().split(indices_len)
        for i in indices_split:
            assert i.std() == 0

        xy_split = xy.float().split(indices_len)

        # Compute means (mass center)
        xy_means = [torch.mean(xy, dim=0) for xy in xy_split]

        assert len(xy_means) == len(mean_depths), (
            len(xy_means),
            len(mean_depths),
        )

        points = torch.cat(
            [torch.stack(xy_means), mean_depths],
            dim=1,
        )

        return points

    def _unproject(self, cams: render.Cameras, points_xyd: Tensor) -> Tensor:
        points_xyd_batch = points_xyd[None, :]
        points_world = cams.unproject_points(points_xyd_batch)
        return points_world[0]
