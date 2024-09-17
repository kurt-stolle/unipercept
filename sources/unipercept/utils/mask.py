from __future__ import annotations

import enum as E

import torch
import torch.signal
import torchvision.transforms.v2.functional as tvfn
from torch import Tensor, nn

from unipercept.utils.signal import get_gaussian_2d

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
    axes = torch.stack(_get_index_axes_like(masks, stride=stride), dim=-1)

    if use_vmap:
        return torch.vmap(_get_mass_center, (None, -1), (1))(masks, axes)
    cmap = []
    for i in range(axes.shape[-1]):
        cmap.append(_get_mass_center(masks, axes[..., i]))
    return torch.stack(cmap, dim=-1)


@torch.no_grad()
def masks_to_boxes(
    masks: Tensor,
    stride: int = 1,
    filter_size: int | tuple[int, int] | None = None,
    threshold: float = 0.5,
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
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    if masks.ndim != 3:
        msg = f"Expected masks to have 3 dimensions, got {masks.ndim}"
        raise ValueError(msg)

    if filter_size is not None and threshold > 0.0:
        masks_blur = masks.float()
        masks_blur = tvfn.gaussian_blur_image(masks_blur, filter_size)
        masks_blur = masks >= threshold
        masks_blur_valid = masks_blur.int().sum(dim=(-1, -2)) > 0
        masks[masks_blur_valid] = masks_blur[masks_blur_valid]

    if masks.dtype != torch.bool:
        masks = masks >= threshold

    axes = _get_index_axes_like(masks, stride=stride)
    return _get_bounding_box_batched(axes, masks)


@torch.no_grad()
def _get_index_axes_like(t: Tensor, *, stride: int) -> Tensor:
    """
    Returns a grid of indices with shape (H, W, 2).
    """
    h, w = t.shape[-2:]
    with t.device:
        y = torch.arange(0, h * stride, stride, dtype=torch.float) + stride // 2
        x = torch.arange(0, w * stride, stride, dtype=torch.float) + stride // 2
    # g = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return torch.meshgrid(y, x, indexing="ij")


def _get_mass_center(masks: Tensor, index_axes: Tensor) -> Tensor:
    sum_pixels = (masks * index_axes).sum(dim=(-1, -2))
    num_pixels = masks.sum(dim=(-1, -2)).clamp(min=1.0)
    return sum_pixels / num_pixels

    # nanmean not yet implemented in Inductor?
    # return (m / m * ax).nanmean(dim=(-2, -1))


def _get_bounding_box_batched(index_axes: Tensor, masks: Tensor):
    y, x = index_axes

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


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

    if ndim == 2 or ndim == 3:
        masks = masks.unsqueeze(dim=0)
    elif ndim > 4:
        masks = masks.reshape((-1,) + shape[-3:])

    fp = torch.is_floating_point(masks)

    k_size_wmax = k_sizes.max().item()
    k_size_hmax = k_size_wmax
    ks_list: list[Tensor] = []
    for k_size in k_sizes.unbind(0):
        k_item = k_size.item()
        sigma = alpha * k_item + eps

        k = get_gaussian_2d(
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
