"""
Visualization utilities.
"""

from __future__ import annotations

import typing as T
import warnings
from xml.etree.ElementTree import PI

import numpy as np
import PIL.Image as pil_image
import torch
import torchvision.transforms.functional as F
import unipercept.data.points as _DP
import unipercept.data.types as _DT

if T.TYPE_CHECKING:
    from matplotlib.axes import Axes as MatplotlibAxesObject
    from PIL.Image import Image as PILImageObject

__all__ = ["draw_image", "draw_image_segmentation", "draw_image_depth"]


@torch.inference_mode()
def plot_input_data(data: _DP.InputData, /, info: _DT.Metadata, height=4) -> T.Any:
    """
    Plots the given input data.
    """

    import matplotlib.pyplot as plt

    if len(data.batch_size) > 0:
        warnings.warn("Received batched input data, plotting only the first element!", stacklevel=2)
        data = data[0]

    caps = data.captures
    nrows = caps.batch_size[0]
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * caps.images.shape[-2] / caps.images.shape[-1], height)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), sharex=True, sharey=True)

    for i, cap in enumerate(caps):
        draw_image(cap.images, ax=axs[i, 0])

        cap = cap.fillna(inplace=False)

        # pan = cap.get("panoptic", -1 * torch.ones((cap.images.shape[-2], cap.images.shape[-1]), dtype=torch.int64))
        draw_image_segmentation(cap.segmentations, info, ax=axs[i, 1])

        # dep = cap.get("depth", torch.zeros((cap.images.shape[-2], cap.images.shape[-1]), dtype=torch.float32))
        draw_image_depth(cap.depths, info, ax=axs[i, 2])
    return fig


@torch.inference_mode()
def draw_image(img: torch.Tensor, /, ax: MatplotlibAxesObject | None = None) -> PILImageObject:
    """
    Shows the given images.
    """

    assert len(img.shape) == 3, f"Expected image with 3HW dimensions, got {img.shape}!"
    assert img.shape[0] == 3, f"Expected image with 3 channels, got {img.shape[0]}!"

    img = F.to_pil_image(img.detach())

    if ax is not None:
        ax.imshow(np.asarray(img))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return img


@torch.inference_mode()
def draw_image_segmentation(
    pan: _DP.PanopticMap | torch.Tensor,
    /,
    info: _DT.Metadata,
    ax: MatplotlibAxesObject | None = None,
    scale=1,
    **kwargs,
) -> PILImageObject:
    """
    Draws the panoptic map using the given info metadata or a color scheme generated ad-hoc.
    """
    from ._visualizer import Visualizer

    assert len(pan.shape) == 2, f"Expected image with HW dimensions, got {pan.shape}!"

    pan = pan.detach().as_subclass(_DP.PanopticMap)

    img = torch.zeros((3, pan.shape[-2], pan.shape[-1]), dtype=torch.float32)
    vis = Visualizer(torch.zeros_like(img), info, scale=scale)
    vis.draw_segmentation(pan, alpha=1.0, **kwargs)

    out = vis.output.get_image()

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return pil_image.fromarray(out)


@torch.inference_mode()
def draw_image_depth(
    dep: torch.Tensor, /, info: _DT.Metadata, palette: str = "viridis", ax: MatplotlibAxesObject | None = None
) -> PILImageObject:
    """
    Draws the depth map as an RGB heatmap, normalized from 0 until 1 using the given ``'max_depth'`` in the ``info``
    parameter, and then mapped to a color scheme generated ad-hoc and expressed as uint8.
    """

    import seaborn as sns

    assert len(dep.shape) == 2, f"Expected image with HW dimensions, got {dep.shape}!"

    # dis_max = 1 / info.depth_max
    dep = (1 / dep) / info.depth_max

    dep = dep.detach()
    dep.clamp_(0, 1).numpy()

    cmap = sns.color_palette(palette, as_cmap=True)
    out = cmap(dep)

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return pil_image.fromarray(np.uint8(out * 255))
