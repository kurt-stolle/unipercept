"""
Visualization utilities. This package is only loaded if ``matplotlib`` and ``seaborn`` are installed.
"""

from __future__ import annotations

import typing as T
import warnings

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_image
import torch
import torchvision.transforms.functional as F
from matplotlib import colors
from matplotlib.axes import Axes as MatplotlibAxesObject
from PIL.Image import Image as PILImageObject
from tensordict import TensorDict, TensorDictBase

from unipercept.data.sets import Metadata
from unipercept.data.tensors import PanopticMap
from unipercept.log import logger
from unipercept.model import InputData, ModelOutput

__all__ = [
    "plot_input_data",
    "plot_predictions",
    "plot_depth_error",
    "draw_image",
    "draw_image_segmentation",
    "draw_image_depth",
    "draw_image_heatmap",
    "draw_map",
    "draw_layers",
]

from unipercept.utils.missing import MissingValue

SKIP = MissingValue("SKIP")


def plot_input_data(
    data: InputData | T.Iterable[InputData] | TensorDictBase | T.Mapping[str, T.Any],
    /,
    info: Metadata,
    height: float = 4.0,
    scale: float = 1.0,
    image_options: MissingValue[SKIP] | T.Mapping[str, T.Any] | None = None,
    segmentation_options: T.Mapping[str, T.Any] | None = None,
    depth_options: T.Mapping[str, T.Any] | None = None,
    title: str | None = None,
) -> T.Any:
    """
    Plots input data from the given inputdata object.
    """
    from unipercept.model import InputData

    if not isinstance(data, InputData):
        # Mapping must first be converted to a TensorDict
        if isinstance(data, T.Mapping):
            data = TensorDict.from_dict(data)
        # Handle non-InputData types or raise an error
        if isinstance(data, TensorDictBase):
            data = InputData.from_tensordict(data)
        elif isinstance(data, T.Iterable):
            assert all(isinstance(d, InputData) for d in data), data
            data = torch.stack([d.fillna(inplace=False) for d in data])  # type: ignore
        else:
            msg = f"Cannot plot data type: {type(data)}"
            return TypeError(msg)
    if data.batch_dims > 1:
        msg = f"Invalid data shape: {data.batch_size}"
        return ValueError(msg)
    if data.batch_dims == 0:
        data = data.unsqueeze(0)

    n_batch = data.batch_size[0]
    n_caps = data.captures.batch_size[-1]

    h_caps, w_caps = data.captures.images.shape[-2:]

    nrows = n_batch * n_caps
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * h_caps / w_caps, height)

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

    for b, data_single in enumerate(data.unbind(0)):
        for i, cap in enumerate(data_single.captures.unbind(0)):
            row = axs[b * n_caps + i, :]
            row[0].set_ylabel(f"Batch {b+1} Frame {i+1}")

            if image_options is None:
                image_options = {}
            if image_options not in SKIP:
                draw_image(cap.images, ax=row[0], scale=scale, **image_options)

            if segmentation_options is None:
                segmentation_options = {}
            if segmentation_options not in SKIP:
                segmentation_options.setdefault(
                    "depth_map", cap.depths / info.depth_max
                )
                draw_image_segmentation(
                    cap.segmentations,
                    info,
                    ax=row[1],
                    scale=scale,
                    **segmentation_options,
                )

            if depth_options is None:
                depth_options = {}
            if depth_options not in SKIP:
                draw_image_depth(
                    cap.depths, info, ax=row[2], scale=scale, **depth_options
                )

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout(pad=0)
    return fig


def plot_predictions(
    inputs: InputData,
    predictions: T.Mapping[str, Tensor] | ModelOutput,
    /,
    info: Metadata,
    height: float = 4.0,
    scale: float = 1.0,
    image_options: MissingValue[SKIP] | dict[str, T.Any] | None = None,
    segmentation_options: dict[str, T.Any] | MissingValue[SKIP] | None = None,
    segmentation_key: str = "panoptic_segmentation",
    depth_options: dict[str, T.Any] | MissingValue[SKIP] | None = None,
    depth_key="depth",
) -> T.Any:
    """
    Plots the given input data.
    """

    if isinstance(predictions, ModelOutput):
        predictions = predictions.predictions

    img = inputs.captures.images[
        :, 0, :, :, :
    ]  # batch (=1) x pairs (=1) x C (=3) x H (=1024) x W (=2048)
    seg = [
        p.get(segmentation_key, None) for p in predictions
    ]  # batch (=1) x H (=1024) x W (=2048)
    if all(s is None for s in seg):
        seg = None
        segmentation_options = SKIP
    dep = [
        p.get(depth_key, None) for p in predictions
    ]  # batch (=1) x H (=1024) x W (=2048))
    if all(d is None for d in dep):
        dep = None
        depth_options = SKIP

    nrows = img.size(0)
    ncols = 3  # image, segmentation, depth
    figsize = (3 * height * img.shape[-2] / img.shape[-1], height)

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
            draw_image(img[i], ax=axs[i, 0], scale=scale, **image_options)

        if segmentation_options is None:
            segmentation_options = {}
        if segmentation_options not in SKIP:
            if dep is not None:
                segmentation_options.setdefault("depth_map", dep[i] / info.depth_max)
            draw_image_segmentation(
                seg[i], info, ax=axs[i, 1], scale=scale, **segmentation_options
            )

        if depth_options is None:
            depth_options = {}
        if depth_options not in SKIP:
            draw_image_depth(dep[i], info, ax=axs[i, 2], scale=scale, **depth_options)

    fig.tight_layout(pad=0)
    return fig


def plot_depth_error(
    pred: torch.Tensor,
    true: torch.Tensor,
    info: Metadata,
    *,
    palette: str = "viridis",
    relative: bool = True,
    scale: float = 1.0,
) -> T.Any:
    """
    Draws the depth map as an RGB heatmap, normalized from 0 until 1 using the given ``'max_depth'`` in the ``info``
    parameter, and then mapped to a color scheme generated ad-hoc and expressed as uint8.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    true = _squeeze_to_2d(true.detach().cpu())
    pred = _squeeze_to_2d(pred.detach().cpu())
    error = (true - pred).abs()
    valid = true > 0
    error[~valid] = 0
    if relative:
        error[valid] /= true[valid]
    fig, ax = plt.subplots()

    divisor = make_axes_locatable(ax)
    cax = divisor.append_axes("right", size="5%", pad=0.05)

    cim = ax.imshow(error, cmap=palette)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(cim, cax=cax, orientation="vertical")

    dax = divisor.append_axes("bottom", size="15%", pad=0.05)
    dax.hist(
        error[valid].flatten(),
        color="red",
        alpha=0.9,
        bins=round(info.depth_max),
        density=True,
    )
    dax.set_xlabel("Error distribution")
    dax.set_ylabel("")
    dax.set_yticks([])

    fig.tight_layout(pad=0)
    return fig


def draw_image(
    img: torch.Tensor, /, ax: MatplotlibAxesObject | None = None, scale: float = 1.0
) -> PILImageObject:
    """
    Shows the given images.
    """

    while img.ndim > 3 and img.shape[0] == 1:
        img = img.squeeze(0)

    assert len(img.shape) == 3, f"Expected image with CHW dimensions, got {img.shape}!"
    assert img.shape[0] in (
        1,
        3,
    ), f"Expected image with 1 or 3 channels, got {img.shape[0]}!"

    img_pil = F.to_pil_image(img.detach().cpu())
    img_pil = scale_pil_image(img_pil, scale)

    if ax is not None:
        ax.imshow(img_pil)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return img_pil


def draw_image_segmentation(
    pan: PanopticMap | torch.Tensor,
    info: Metadata,
    *,
    ax: MatplotlibAxesObject | None = None,
    scale: float = 1.0,
    **kwargs,
) -> PILImageObject:
    """
    Draws the panoptic map using the given info metadata or a color scheme generated ad-hoc.
    """
    from unipercept.data.tensors import PanopticMap

    from ._visualizer import Visualizer

    while pan.ndim > 2 and pan.shape[0] == 1:
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
    *,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
    background: torch.Tensor | None = None,
    scale: float = 1.0,
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

        out_list.append(pil_image.fromarray(floating_to_uint8(out_layer)))

    out = out_list.pop()
    for out_layer in out_list:
        out = pil_image.alpha_composite(out, out_layer)
    out = scale_pil_image(out, scale)
    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return rgba_to_rgb(out, background=background)


def draw_map(
    map: torch.Tensor,
    *,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
    background: torch.Tensor | None = None,
    scale: float = 1.0,
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
    out = floating_to_pil(out)
    out = scale_pil_image(out, scale)

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return rgba_to_rgb(out, background=background)


def draw_image_depth(
    dep: torch.Tensor,
    info: Metadata,
    *,
    palette: str = "viridis",
    ax: MatplotlibAxesObject | None = None,
    scale: float = 1.0,
) -> PILImageObject:
    """
    Draws the depth map as an RGB heatmap, normalized from 0 until 1 using the given ``'max_depth'`` in the ``info``
    parameter, and then mapped to a color scheme generated ad-hoc and expressed as uint8.
    """

    import seaborn as sns

    dep = _squeeze_to_2d(dep.detach())
    dep = dep / info.depth_max
    dep = dep.clamp(0, 1)

    cmap = sns.color_palette(palette, as_cmap=True)

    out = cmap(dep.numpy())

    out = floating_to_pil(out)
    out = scale_pil_image(out, scale)

    if ax is not None:
        ax.imshow(out)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return rgba_to_rgb(out)


def draw_image_heatmap(
    heatmap: torch.Tensor,
    *,
    palette: str = "plasma",
    ax: MatplotlibAxesObject | None = None,
    scale: float = 1.0,
    normalize: bool = True,
) -> PILImageObject:
    """
    Draws a map of values [0,1] as an RGB heatmap.
    """

    import seaborn as sns

    heatmap = _squeeze_to_2d(heatmap.detach())
    hmin = heatmap.min().item()
    hmax = heatmap.max().item()
    if normalize:
        out = (heatmap - hmin) / (hmax - hmin)
    elif hmin < 0 or hmax > 1:
        logger.warning(f"Drawing heatmap with values outside [0,1]: {hmin} to {hmax}")
        out = heatmap.clamp(0, 1)
    else:
        out = heatmap
    cmap = sns.color_palette(palette, as_cmap=True)
    out = cmap(heatmap.numpy())

    out = floating_to_pil(out)
    out = scale_pil_image(out, scale)

    if ax is not None:
        if not normalize:
            ax.imshow(heatmap, cmap=palette, vmin=0, vmax=1)
        else:
            plt.colorbar(ax.imshow(heatmap, cmap=palette, vmin=hmin, vmax=hmax))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return rgba_to_rgb(out)


def _squeeze_to_2d(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    while t.ndim > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2:
        msg = f"Expected 2D tensor, got {t.shape}"
        raise ValueError(msg)
    return t


def rgba_to_rgb(
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


def pyplot_to_pil(fig) -> PILImageObject:
    """
    Convert a Matplotlib figure to a PIL Image
    """
    return pil_image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def floating_to_uint8(mat: np.ndarray) -> np.ndarray:
    """
    Convert a float matrix to uint8
    """

    mat = np.asarray(mat)
    assert mat.dtype == np.floating, f"Expected float matrix, got {mat.dtype}!"

    mat = mat * 255.999
    mat = np.clip(mat, 0, 255)

    return mat.astype(np.uint8)


def floating_to_pil(mat: np.ndarray) -> pil_image.Image:
    """
    Convert a float matrix to a PIL image
    """
    try:
        return F.to_pil_image(mat)
    except Exception as e:  # noqa: PIE786
        warnings.warn(f"Failed to convert to PIL image: {e}", stacklevel=2)
        return pil_image.fromarray(floating_to_uint8(mat))


def scale_pil_image(img: PILImageObject, scale: float) -> PILImageObject:
    """
    Scales a PIL image by the given factor.
    """
    return img.resize((round(img.width * scale), round(img.height * scale)))
