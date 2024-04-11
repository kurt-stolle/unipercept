from __future__ import annotations
import math
import typing as T
import torch
import enum as E

from torch import Tensor
import torch.signal
import torch.nn as nn
import torchvision.transforms.v2.functional as tvfn

__all__ = ["masks_to_centers", "masks_to_boxes"]


def masks_to_centers(masks: Tensor, stride: int = 1, use_vmap: bool = False) -> Tensor:
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

    if use_vmap:
        return torch.vmap(_get_mass_center, (None, -1), (1))(masks, axes)
    else:
        cmap = []
        for i in range(axes.shape[-1]):
            cmap.append(_get_mass_center(masks, axes[..., i]))
        return torch.stack(cmap, dim=-1)


@torch.no_grad()
def masks_to_boxes(
    masks: Tensor,
    stride: int = 1,
    use_vmap: bool = False,
    filter_size: int | T.Tuple[int, int] | None = 15,
    filter_threshold: float = 0.5,
) -> Tensor:
    """
    Convert masks to bounding boxes.

    Parameters
    ----------
    masks
        Mask tensor with shape (N, H, W)
    stride
        Scaling factor of the mask.
    filter_size
        Size of the kernel used to blur the mask. By default the size is 15.
    filter_threshold
        Threshold for the filter. By default the threshold is 0.33.

    Returns
    -------
        Bounding box tensor (N, (X1,Y1,X2,Y2)), where X in [0,W] and Y in [0,H]
    """
    masks = masks.bool().permute(0, 2, 1)  # Ensure output is XY not Y
    if filter_size is not None and filter_threshold > 0.0:
        masks_blur = masks.float()
        masks_blur = tvfn.gaussian_blur_image(masks_blur, filter_size)
        masks_blur = masks > filter_threshold
        masks_blur_valid = masks_blur.int().sum(dim=(-1, -2)) > 0
        masks[masks_blur_valid] = masks_blur[masks_blur_valid]
    masks = masks.long()

    axes = _get_index_axes_like(masks, stride=stride)

    if use_vmap:
        xyxy = torch.vmap(_get_bounding_box, (-1, None), (1, 1))(axes, masks)
        return torch.cat(xyxy, dim=-1)
    else:
        xy1 = []
        xy2 = []
        for i in range(axes.shape[-1]):
            min_index, max_index = _get_bounding_box(axes[..., i], masks=masks)
            xy1.append(min_index)
            xy2.append(max_index)

        return torch.cat([torch.stack(xy1, dim=-1), torch.stack(xy2, dim=-1)], dim=-1)


def _get_index_axes_like(t: Tensor, *, stride: int) -> Tensor:
    """
    Returns a grid of indices with shape (H, W, 2).
    """
    with torch.no_grad():
        h, w = t.shape[-2:]
        y = torch.arange(0, h * stride, stride, dtype=torch.float, device=t.device)
        x = torch.arange(0, w * stride, stride, dtype=torch.float, device=t.device)
        g = (
            torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1) + stride // 2
        )  # (H, W, 2)
    return g


def _get_mass_center(masks: Tensor, index_axes: Tensor) -> Tensor:
    sum_pixels = (masks * index_axes).sum(dim=(-1, -2))
    num_pixels = masks.sum(dim=(-1, -2)).clamp(min=1.0)
    return sum_pixels / num_pixels

    # nanmean not yet implemented in Inductor?
    # return (m / m * ax).nanmean(dim=(-2, -1))


def _get_bounding_box(index_axes: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    # Select the maximum index in the valid indices tensor
    max_coords = masks * index_axes[None, :, :]
    max_index = max_coords.amax(dim=(-1, -2))

    # Apply a mask of the maximum value to the invalid indices
    min_coords = max_coords + masks.eq(0) * max_index[:, None, None]
    min_index = min_coords.amin(dim=(-1, -2))

    return min_index, max_index


class BlurSizeMethod(E.StrEnum):
    SHORT = E.auto()
    LONG = E.auto()
    MEAN = E.auto()


def blur_masks(
    masks: Tensor,
    alpha: float = 0.15,
    eps: float = 0.35,
    k_factor: float = 1.0,
    k_method: BlurSizeMethod = BlurSizeMethod.SHORT,
) -> Tensor:
    if masks.numel() == 0:
        return masks

    dtype = masks.dtype
    ndim = masks.ndim
    if ndim == 2:
        masks = masks.unsqueeze(dim=0)
    shape = masks.shape

    ltbr = masks_to_boxes(masks)
    w = ltbr[:, 2] - ltbr[:, 0]
    h = ltbr[:, 3] - ltbr[:, 1]

    match k_method:
        case BlurSizeMethod.SHORT:
            k_sizes = torch.min(w, h)
        case BlurSizeMethod.LONG:
            k_sizes = torch.max(w, h)
        case BlurSizeMethod.MEAN:
            k_sizes = (w + h) / 2
        case _:
            msg = f"Invalid blur size method: {k_method}"
            raise ValueError(msg)

    k_sizes = (k_sizes * k_factor).int()
    k_sizes = k_sizes // 2 * 2 + 1
    k_sizes = k_sizes.clamp(3)

    if ndim == 2:
        masks = masks.unsqueeze(dim=0)
    elif ndim == 3:
        masks = masks.unsqueeze(dim=0)
    elif ndim > 4:
        masks = masks.reshape((-1,) + shape[-3:])

    fp = torch.is_floating_point(masks)

    k_size_wmax = k_sizes.max().item()
    k_size_hmax = k_size_wmax
    ks_list: T.List[Tensor] = []
    for k_size in k_sizes.unbind(0):
        k_item = k_size.item()
        sigma = alpha * k_item + eps

        k = _get_gaussian_kernel2d(
            [k_item, k_item],
            [sigma, sigma],
            dtype if fp else torch.float32,
            device=masks.device,
        )

        k_size_wdiff = k_size_wmax - k_item
        k_size_hdiff = k_size_hmax - k_item
        k = torch.nn.functional.pad(
            k,
            [
                k_size_wdiff // 2,
                k_size_wdiff // 2,
                k_size_hdiff // 2,
                k_size_hdiff // 2,
            ],
        )
        ks_list.append(k)

    ks = torch.stack(ks_list, dim=0).unsqueeze(1)

    assert ks.ndim == 4, ks.shape  # (N, 1, K, K)

    output = masks if fp else masks.to(dtype=torch.float32)

    # padding = (left, right, top, bottom)
    padding = [
        k_size_wmax // 2,
        k_size_wmax // 2,
        k_size_hmax // 2,
        k_size_hmax // 2,
    ]
    output = nn.functional.pad(output, padding, mode="reflect")
    output = nn.functional.conv2d(output, ks, groups=shape[-3])

    if ndim == 2:
        output = output.squeeze(dim=0).squeeze(dim=0)
    elif ndim == 3:
        output = output.squeeze(dim=0)
    elif ndim > 4:
        output = output.reshape(shape)

    if not fp:
        output = output.round_().to(dtype=dtype)

    return output


def _get_gaussian_kernel2d(
    kernel_size: T.List[int],
    sigma: T.List[float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    with torch.no_grad(), device:
        k_x = torch.signal.windows.gaussian(
            kernel_size[0], std=sigma[0], sym=True, dtype=dtype
        )
        k_y = torch.signal.windows.gaussian(
            kernel_size[1], std=sigma[1], sym=True, dtype=dtype
        )
        k = k_y.unsqueeze(-1) * k_x
    return k


if __name__ == "__main__":
    # Example usage, this draws a bounding box around the masks and a circle at the
    # center of the mask.
    import matplotlib.patches as pat
    import matplotlib.pyplot as plt
    import torch

    import unipercept as up

    mask1 = torch.zeros(128, 128)
    mask1[32:96, 32:96] = 1

    mask2 = torch.zeros(128, 128)
    mask2[64:76, 64:120] = 1

    masks = torch.stack([torch.zeros_like(mask1), mask1, mask2])
    boxes = up.utils.mask.masks_to_boxes(masks)
    centers = up.utils.mask.masks_to_centers(masks)

    fig, ax = plt.subplots()
    ax.imshow(masks.argmax(0))

    for xyxy in boxes.unbind(0):
        x1, y1, x2, y2 = map(float, xyxy.tolist())
        w = x2 - x1
        h = y2 - y1

        ax.add_patch(pat.Rectangle((x1, y1), w, h, fill=False, edgecolor="red"))

    for xy in centers:
        x, y = map(int, xy.tolist())
        ax.add_patch(pat.Circle((x, y), radius=2, fill=True, color="red"))
