"""
Visualization utilities.
"""

from __future__ import annotations

import typing as T
import warnings

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_image
import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict, TensorDictBase

if T.TYPE_CHECKING:
    from matplotlib.axes import Axes as MatplotlibAxesObject
    from PIL.Image import Image as PILImageObject

    from unipercept.data.sets import Metadata
    from unipercept.data.tensors import PanopticMap
    from unipercept.model import InputData, ModelOutput

__all__ = ["draw_image", "draw_image_segmentation", "draw_image_depth"]

from unicore.utils.missing import MissingValue

SKIP = MissingValue("SKIP")


def plot_input_data(
    data: InputData,
    /,
    info: Metadata,
    height=4,
    image_options: MissingValue["SKIP"] | T.Optional[dict[str, T.Any]] = None,
    segmentation_options: T.Optional[dict[str, T.Any]] = None,
    depth_options: T.Optional[dict[str, T.Any]] = None,
) -> T.Any:
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

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), sharex=True, sharey=True, squeeze=False
    )

    for i, lbl in enumerate(("Image", "Segmentation", "Depth")):
        axs[-1, i].set_xlabel(lbl)

    for i, cap in enumerate(caps):
        axs[i, 0].set_ylabel(f"Frame {i+1}")

        if image_options is None:
            image_options = {}
        if image_options not in SKIP:
            draw_image(cap.images, ax=axs[i, 0], **image_options)

        cap = cap.fillna(inplace=False)

        if segmentation_options is None:
            segmentation_options = {}
        if segmentation_options not in SKIP:
            segmentation_options.setdefault("depth_map", cap.depths / cap.depths.max())
            draw_image_segmentation(cap.segmentations, info, ax=axs[i, 1], **segmentation_options)

        if depth_options is None:
            depth_options = {}
        if depth_options not in SKIP:
            draw_image_depth(cap.depths, info, ax=axs[i, 2], **depth_options)

    fig.tight_layout(pad=0)
    return fig


def plot_predictions(
    inputs: InputData,
    predictions: TensorDictBase | ModelOutput,
    /,
    info: Metadata,
    height=4,
    image_options: MissingValue["SKIP"] | T.Optional[dict[str, T.Any]] = None,
    segmentation_options: T.Optional[dict[str, T.Any]] = None,
    depth_options: T.Optional[dict[str, T.Any]] = None,
) -> T.Any:
    """
    Plots the given input data.
    """

    try:
        predictions: TensorDict = predictions.get("predictions")
    except (AttributeError, KeyError):
        pass

    img = inputs.captures.images[:, 0, :, :, :]  # batch (=1) x pairs (=1) x C (=3) x H (=1024) x W (=2048)
    seg = predictions.get("segmentations")  # batch (=1) x H (=1024) x W (=2048)
    dep = predictions.get("depths")  # batch (=1) x H (=1024) x W (=2048))

    nrows = predictions.batch_size[0]
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * seg.shape[-2] / seg.shape[-1], height)

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), sharex=True, sharey=True, squeeze=False
    )

    for i, lbl in enumerate(("Input", "Segmentation", "Depth")):
        axs[-1, i].set_xlabel(lbl)

    for i in range(nrows):
        axs[i, 0].set_ylabel(f"Frame {i+1}")

        if image_options is None:
            image_options = {}
        if image_options not in SKIP:
            draw_image(img[i], ax=axs[i, 0], **image_options)

        if segmentation_options is None:
            segmentation_options = {}
        if segmentation_options not in SKIP:
            segmentation_options.setdefault("depth_map", dep[i] / dep[i].max())
            draw_image_segmentation(seg[i], info, ax=axs[i, 1], **segmentation_options)

        if depth_options is None:
            depth_options = {}
        if depth_options not in SKIP:
            draw_image_depth(dep[i], info, ax=axs[i, 2], **depth_options)

    fig.tight_layout(pad=0)
    return fig


def draw_image(img: torch.Tensor, /, ax: MatplotlibAxesObject | None = None) -> PILImageObject:
    """
    Shows the given images.
    """

    if img.ndim > 3:
        img = img.squeeze(0)

    assert len(img.shape) == 3, f"Expected image with CHW dimensions, got {img.shape}!"
    assert img.shape[0] in (1, 3), f"Expected image with 1 or 3 channels, got {img.shape[0]}!"

    img = F.to_pil_image(img.detach())

    if ax is not None:
        ax.imshow(np.asarray(img))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return img


def draw_image_segmentation(
    pan: PanopticMap | torch.Tensor,
    /,
    info: Metadata,
    ax: MatplotlibAxesObject | None = None,
    scale=1,
    **kwargs,
) -> PILImageObject:
    """
    Draws the panoptic map using the given info metadata or a color scheme generated ad-hoc.
    """
    from unipercept.data.tensors import PanopticMap

    from ._visualizer import Visualizer

    assert len(pan.shape) == 2, f"Expected image with HW dimensions, got {pan.shape}!"

    pan = pan.detach().as_subclass(PanopticMap)

    img = torch.zeros((3, pan.shape[-2], pan.shape[-1]), dtype=torch.float32)
    vis = Visualizer(torch.zeros_like(img), info, scale=scale)
    vis.draw_segmentation(pan, alpha=1.0, **kwargs)

    out = vis.output.get_image()

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return pil_image.fromarray(out)


def draw_image_depth(
    dep: torch.Tensor, /, info: Metadata, palette: str = "viridis", ax: MatplotlibAxesObject | None = None
) -> PILImageObject:
    """
    Draws the depth map as an RGB heatmap, normalized from 0 until 1 using the given ``'max_depth'`` in the ``info``
    parameter, and then mapped to a color scheme generated ad-hoc and expressed as uint8.
    """

    import seaborn as sns

    if dep.ndim > 2:
        dep = dep.squeeze(0)
    if dep.ndim > 2:
        dep = dep.squeeze_(0)

    assert dep.ndim == 2, f"Expected image with HW dimensions, got {dep.shape}!"

    dep = dep.detach() / info.depth_max
    dep.clamp_(0, 1).numpy()

    cmap = sns.color_palette(palette, as_cmap=True)
    out = cmap(dep)

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return pil_image.fromarray(np.uint8(out * 255))
