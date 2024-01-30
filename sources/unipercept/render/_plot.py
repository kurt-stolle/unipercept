"""
Visualization utilities. This package is only loaded if ``matplotlib`` and ``seaborn`` are installed.
"""

from __future__ import annotations

import typing as T
import warnings

import matplotlib.colors as colors
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

__all__ = [
    "plot_input_data",
    "plot_predictions",
    "draw_image",
    "draw_image_segmentation",
    "draw_image_depth",
    "draw_map",
    "draw_layers",
]

from unipercept.utils.missing import MissingValue

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
        warnings.warn(
            "Received batched input data, plotting only the first element!",
            stacklevel=2,
        )
        data = data[0]

    caps = data.captures
    nrows = caps.batch_size[0]
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * caps.images.shape[-2] / caps.images.shape[-1], height)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
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
            draw_image_segmentation(
                cap.segmentations, info, ax=axs[i, 1], **segmentation_options
            )

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
    image_options: MissingValue["SKIP"] | dict[str, T.Any] | None = None,
    segmentation_options: dict[str, T.Any] | MissingValue["SKIP"] | None = None,
    depth_options: dict[str, T.Any] | MissingValue["SKIP"] | None = None,
) -> T.Any:
    """
    Plots the given input data.
    """

    try:
        predictions: TensorDict = predictions.get("predictions")
    except (AttributeError, KeyError):
        pass

    img = inputs.captures.images[
        :, 0, :, :, :
    ]  # batch (=1) x pairs (=1) x C (=3) x H (=1024) x W (=2048)
    seg = predictions.get("segmentations", None)  # batch (=1) x H (=1024) x W (=2048)
    if seg is None:
        segmentation_options = SKIP
    dep = predictions.get("depths", None)  # batch (=1) x H (=1024) x W (=2048))
    if dep is None:
        depth_options = SKIP

    nrows = predictions.batch_size[0]
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * seg.shape[-2] / seg.shape[-1], height)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
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
            if dep is not None:
                segmentation_options.setdefault("depth_map", dep[i] / dep[i].max())
            draw_image_segmentation(seg[i], info, ax=axs[i, 1], **segmentation_options)

        if depth_options is None:
            depth_options = {}
        if depth_options not in SKIP:
            draw_image_depth(dep[i], info, ax=axs[i, 2], **depth_options)

    fig.tight_layout(pad=0)
    return fig


def draw_image(
    img: torch.Tensor, /, ax: MatplotlibAxesObject | None = None
) -> PILImageObject:
    """
    Shows the given images.
    """

    if img.ndim > 3:
        img = img.squeeze(0)

    assert len(img.shape) == 3, f"Expected image with CHW dimensions, got {img.shape}!"
    assert img.shape[0] in (
        1,
        3,
    ), f"Expected image with 1 or 3 channels, got {img.shape[0]}!"

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

    pan = pan.squeeze(0)

    assert len(pan.shape) == 2, f"Expected image with HW dimensions, got {pan.shape}!"

    pan = pan.detach().as_subclass(PanopticMap)

    img = torch.zeros((3, pan.shape[-2], pan.shape[-1]), dtype=torch.float32)
    vis = Visualizer(torch.zeros_like(img), info, scale=scale)
    vis.draw_segmentation(pan, alpha=1.0, **kwargs)

    out = vis.output.get_image()

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return pil_image.fromarray(out).convert("RGB")


def draw_layers(
    layers: torch.Tensor,
    /,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
    background: torch.Tensor | None = None,
) -> PILImageObject:
    """
    Draw a layered tensor by rendering each layer as a colored heatmap.

    Parameters
    ----------
    layers : torch.Tensor
        The tensor to draw, with dimensions (..., L, H, W), where L is the amount of layers.
    palette : str, optional
        The name of the palette to use, by default "viridis". Each heatmap layer will use a different color from the
        palette.
    ax : MatplotlibAxesObject, optional
        The axes to draw to, by default None.
    Returns
    -------
    PILImageObject
        The rendered image.
    """

    import seaborn as sns

    layers = layers.detach()

    for _ in range(max(0, layers.ndim - 3)):
        layers.squeeze_(0)

    if layers.ndim != 3:
        raise ValueError(f"Expected map with (1...)LHW dimensions, got {layers.shape}!")

    # Normalize all layers to floats between 0 and 1
    layers = (layers - layers.min()) / (layers.max() - layers.min())
    layers.clamp_(0, 1)

    # Get the desired colormap
    cmap: colors.Colormap = sns.color_palette(
        palette, n_colors=layers.shape[0], as_cmap=True
    )

    # Allocate the output RGB image
    out_list: list[PILImageObject] = []
    one = np.ones((layers.shape[-2], layers.shape[-1]), dtype=np.float32())

    # Overlay each layer on top of the output image
    for i, layer in enumerate(layers):
        out_layer = cmap(one * ((i + 1) / layers.shape[0]), layer.numpy())
        # out_layer[3] *= layer.numpy()

        out_list.append(pil_image.fromarray(_float_to_uint8(out_layer)))

    out = out_list.pop()
    for out_layer in out_list:
        out = pil_image.alpha_composite(out, out_layer)

    if ax is not None:
        ax.imshow(np.frombuffer(out))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return _rgba_to_rgb(out, background=background)


def draw_map(
    map: torch.Tensor,
    /,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
    background: torch.Tensor | None = None,
) -> PILImageObject:
    """
    Draws a map directly from a tensor, without using the Visualizer class.
    """
    import seaborn as sns

    map = map.detach()

    for _ in range(max(0, map.ndim - 2)):
        map.squeeze_(0)

    if map.ndim != 2:
        raise ValueError(f"Expected map with (1...)HW dimensions, got {map.shape}!")

    # Normalize the map to floats between 0 and 1
    map = (map - map.min()) / (map.max() - map.min())
    map.clamp_(0, 1)

    # Convert to RGB
    cmap = sns.color_palette(palette, as_cmap=True)
    out = cmap(map.numpy())

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    out = pil_image.fromarray(_float_to_uint8(out))
    return _rgba_to_rgb(out, background=background)


def draw_image_depth(
    dep: torch.Tensor,
    /,
    info: Metadata,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
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
    dep.clamp_(0, 1)

    cmap = sns.color_palette(palette, as_cmap=True)
    out = cmap(dep.numpy())

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    out = pil_image.fromarray(_float_to_uint8(out))
    return _rgba_to_rgb(out)


def _rgba_to_rgb(
    img: PILImageObject, background: PILImageObject | torch.Tensor | None = None
) -> PILImageObject:
    """
    Convert an RGBA image to RGB by removing the alpha channel
    """
    from torchvision.transforms.v2.functional import to_pil_image

    out = pil_image.new("RGB", img.size, (0, 0, 0))
    if background is not None:
        if isinstance(background, torch.Tensor):
            background = to_pil_image(background)
        assert isinstance(background, pil_image.Image), f"{type(background)=}"
        out.paste(background.resize(img.size, resample=pil_image.Resampling.BILINEAR))
    out.paste(img, mask=img.split()[3])  # 3 is the alpha channel

    return out


def _figure_to_pil(fig) -> PILImageObject:
    """
    Convert a Matplotlib figure to a PIL Image
    """
    return pil_image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def _float_to_uint8(mat) -> np.ndarray:
    """
    Convert a float matrix to uint8
    """

    mat = mat * 255.999
    mat = np.clip(mat, 0, 255)

    return mat.astype(np.uint8)
