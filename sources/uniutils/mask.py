import torch

__all__ = ["masks_to_centers", "masks_to_boxes"]


@torch.jit.script_if_tracing
def masks_to_centers(masks: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Converts a mask to a center point.

    Parameters
    ----------
    masks
        Mask tensor with shape (N, H, W)
    stride
        Scaling factor of the mask.

    Returns
    -------
        Center point tensor (N, (X,Y)), where X in [0,W] and Y in [0,H]
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device, dtype=masks.dtype)

    masks = masks.permute(0, 2, 1)  # Ensure output is XY not YX
    axes = _get_index_axes_like(masks, stride=stride)
    cmap = torch.vmap(_get_mass_center, (None, -1), (1))(masks, axes)

    return cmap


def masks_to_boxes(masks: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Convert masks to bounding boxes.

    Parameters
    ----------
    masks
        Mask tensor with shape (N, H, W)
    stride
        Scaling factor of the mask.

    Returns
    -------
        Bounding box tensor (N, (X1,Y1,X2,Y2)), where X in [0,W] and Y in [0,H]
    """
    # if masks.numel() == 0:
    #     return torch.zeros((0, 4), device=masks.device, dtype=masks.dtype)

    masks = masks.bool().permute(0, 2, 1).long()  # Ensure output is XY not YX
    axes = _get_index_axes_like(masks, stride=stride)
    xyxy = torch.vmap(_get_bounding_box, (-1,), (1, 1))(axes, masks=masks)

    return torch.cat(xyxy, dim=-1)


def _get_index_axes_like(t: torch.Tensor, *, stride: int) -> torch.Tensor:
    """
    Returns a grid of indices with shape (H, W, 2).
    """
    with torch.no_grad():
        h, w = t.shape[-2:]
        y = torch.arange(0, h * stride, stride, dtype=torch.float, device=t.device)
        x = torch.arange(0, w * stride, stride, dtype=torch.float, device=t.device)
        g = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1) + stride // 2  # (H, W, 2)
    return g


def _get_mass_center(masks: torch.Tensor, index_axes: torch.Tensor) -> torch.Tensor:
    sum_pixels = (masks * index_axes).sum(dim=(-1, -2))
    num_pixels = masks.sum(dim=(-1, -2)).clamp(min=1.0)
    return sum_pixels / num_pixels

    # nanmean not yet implemented in Inductor?
    # return (m / m * ax).nanmean(dim=(-2, -1))


def _get_bounding_box(index_axes: torch.Tensor, *, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Select the maximum index in the valid indices tensor
    max_coords = masks * index_axes[None, :, :]
    max_index = max_coords.amax(dim=(-1, -2))

    # Apply a mask of the maximum value to the invalid indices
    min_coords = max_coords + masks.eq(0) * max_index[:, None, None]
    min_index = min_coords.amin(dim=(-1, -2))

    return min_index, max_index
