from __future__ import annotations

import enum as E
import math

import torch
import typing_extensions as TX
from torch import nn, stack, tensor, where, zeros, zeros_like

from unipercept.types import Device, DType, Tensor
from unipercept.utils.check import assert_shape, assert_tensor


class FilterBorder(E.StrEnum):
    CONSTANT = E.auto()
    REFLECT = E.auto()
    REPLICATE = E.auto()
    CIRCULAR = E.auto()


class FilterPadding(E.StrEnum):
    VALID = E.auto()
    SAME = E.auto()


class FilterKind(E.StrEnum):
    CONVOLUTION = "conv"
    CORRELATION = "corr"


def _compute_padding(kernel_size: list[int]) -> list[int]:
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    input: Tensor,
    kernel: Tensor,
    border_type: FilterBorder | str = "reflect",
    normalized: bool = False,
    padding: FilterPadding | str = "same",
    kind: FilterKind | str = "corr",
) -> Tensor:
    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))
    assert_tensor(kernel)
    assert_shape(kernel, ("B", "H", "W"))

    # prepare kernel
    b, c, h, w = input.shape
    if str(kind).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(
            device=input.device, dtype=input.dtype
        )
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = nn.functional.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input: Tensor,
    kernel_x: Tensor,
    kernel_y: Tensor,
    border_type: FilterBorder | str = "reflect",
    normalized: bool = False,
    padding: FilterPadding | str = "same",
) -> Tensor:
    out_x = filter2d(input, kernel_x[..., None, :], border_type, normalized, padding)
    out = filter2d(out_x, kernel_y[..., None], border_type, normalized, padding)
    return out


def filter3d(
    input: Tensor,
    kernel: Tensor,
    border_type: FilterBorder | str = "replicate",
    normalized: bool = False,
) -> Tensor:
    assert_tensor(input)
    assert_shape(input, ("B", "C", "D", "H", "W"))
    assert_tensor(kernel)
    assert_shape(kernel, ("B", "D", "H", "W"))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(
            tmp_kernel
        )

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = nn.functional.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(
        -1,
        tmp_kernel.size(0),
        input_pad.size(-3),
        input_pad.size(-2),
        input_pad.size(-1),
    )

    # convolve the tensor with the kernel.
    output = nn.functional.conv3d(
        input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    return output.view(b, c, d, h, w)


class BlurPool2D(nn.Module):
    r"""Compute blur (anti-aliasing) and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Parameters
    ----------
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.

    Shape
    -----
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    """

    def __init__(self, kernel_size: tuple[int, int] | int, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        self.kernel = torch.as_tensor(
            self.kernel, device=input.device, dtype=input.dtype
        )
        return _blur_pool_by_kernel2d(
            input, self.kernel.repeat((input.shape[1], 1, 1, 1)), self.stride
        )


class MaxBlurPool2D(nn.Module):
    r"""Compute pools and blurs and downsample a given feature map.

    Equivalent to ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```

    See :cite:`zhang2019shiftinvar` for more details.

    Parameters
    ----------
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Shape
    -----
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / stride, W / stride)`

    Returns
    -------
        Tensor: the transformed tensor.

    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        stride: int = 2,
        max_pool_size: int = 2,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        self.kernel = torch.as_tensor(
            self.kernel, device=input.device, dtype=input.dtype
        )
        return _max_blur_pool_by_kernel2d(
            input,
            self.kernel.repeat((input.size(1), 1, 1, 1)),
            self.stride,
            self.max_pool_size,
            self.ceil_mode,
        )


class EdgeAwareBlurPool2D(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        edge_threshold: float = 1.25,
        edge_dilation_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.edge_threshold = edge_threshold
        self.edge_dilation_kernel_size = edge_dilation_kernel_size

    @TX.override
    def forward(self, input: Tensor, epsilon: float = 1e-6) -> Tensor:
        return edge_aware_blur_pool2d(
            input,
            self.kernel_size,
            self.edge_threshold,
            self.edge_dilation_kernel_size,
            epsilon,
        )


def blur_pool2d(
    input: Tensor, kernel_size: tuple[int, int] | int, stride: int = 2
) -> Tensor:
    r"""Compute blurs and downsample a given feature map.

    .. image:: _static/img/blur_pool2d.png

    Parameters
    ----------
        kernel_size: the kernel size for max pooling..
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Shape
    -----
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Returns
    -------
        the transformed tensor.

    Examples
    --------
        >>> input = torch.eye(5)[None, None]
        >>> blur_pool2d(input, 3)
       torch.Tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """
    kernel = get_pascal_kernel_2d(
        kernel_size, norm=True, device=input.device, dtype=input.dtype
    ).repeat((input.size(1), 1, 1, 1))
    return _blur_pool_by_kernel2d(input, kernel, stride)


def max_blur_pool2d(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    stride: int = 2,
    max_pool_size: int = 2,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Compute pools and blurs and downsample a given feature map.

    Parameters
    ----------
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Examples
    --------
        >>> input = torch.eye(5)[None, None]
        >>> max_blur_pool2d(input, 3)
       torch.Tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """
    assert_shape(input, ("B", "C", "H", "W"))

    kernel = get_pascal_kernel_2d(
        kernel_size, norm=True, device=input.device, dtype=input.dtype
    ).repeat((input.shape[1], 1, 1, 1))
    return _max_blur_pool_by_kernel2d(input, kernel, stride, max_pool_size, ceil_mode)


def _blur_pool_by_kernel2d(input: Tensor, kernel: Tensor, stride: int) -> Tensor:
    r"""
    Compute blur_pool by a given :math:`CxC_{out}xNxN` kernel."""
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return nn.functional.conv2d(
        input, kernel, padding=padding, stride=stride, groups=input.shape[1]
    )


def _max_blur_pool_by_kernel2d(
    input: Tensor, kernel: Tensor, stride: int, max_pool_size: int, ceil_mode: bool
) -> Tensor:
    r"""
    Compute max_blur_pool by a given :math:`CxC_(out, None)xNxN` kernel."""
    # compute local maxima
    input = nn.functional.max_pool2d(
        input, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode
    )
    # blur and downsample
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return nn.functional.conv2d(
        input, kernel, padding=padding, stride=stride, groups=input.size(1)
    )


def edge_aware_blur_pool2d(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    edge_threshold: float = 1.25,
    edge_dilation_kernel_size: int = 3,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""Blur the input tensor while maintaining its edges.

    Parameters
    ----------
        input: the input image to blur with shape :math:`(B, C, H, W)`.
        kernel_size: the kernel size for max pooling.
        edge_threshold: positive threshold for the edge decision rule; edge/non-edge.
        edge_dilation_kernel_size: the kernel size for dilating the edges.
        epsilon: for numerical stability.

    Returns
    -------
        The blurred tensor of shape :math:`(B, C, H, W)`.
    """
    assert_shape(input, ("B", "C", "H", "W"))

    input = nn.functional.pad(
        input, (2, 2, 2, 2), mode="reflect"
    )  # pad to avoid artifacts near physical edges
    blurred_input = blur_pool2d(
        input, kernel_size=kernel_size, stride=1
    )  # blurry version of the input

    # calculate the edges (add epsilon to avoid taking the log of 0)
    log_input, log_thresh = (input + epsilon).log2(), (tensor(edge_threshold)).log2()
    edges_x = log_input[..., :, 4:] - log_input[..., :, :-4]
    edges_y = log_input[..., 4:, :] - log_input[..., :-4, :]
    edges_x, edges_y = (
        edges_x.mean(dim=-3, keepdim=True),
        edges_y.mean(dim=-3, keepdim=True),
    )
    edges_x_mask, edges_y_mask = (
        edges_x.abs() > log_thresh.to(edges_x),
        edges_y.abs() > log_thresh.to(edges_y),
    )
    edges_xy_mask = (edges_x_mask[..., 2:-2, :] + edges_y_mask[..., :, 2:-2]).type_as(
        input
    )

    # dilate the content edges to have a soft mask of edges
    dilated_edges = nn.functional.max_pool3d(
        edges_xy_mask, edge_dilation_kernel_size, 1, edge_dilation_kernel_size // 2
    )

    # slice the padded regions
    input = input[..., 2:-2, 2:-2]
    blurred_input = blurred_input[..., 2:-2, 2:-2]

    # fuse the input image on edges and blurry input everywhere else
    blurred = dilated_edges * input + (1.0 - dilated_edges) * blurred_input

    return blurred


def box_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    border_type: FilterBorder | str = "reflect",
    separable: bool = False,
) -> Tensor:
    r"""Blur an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Parameters
    ----------
    image: 
        The image to blur with shape :math:`(B,C,H,W)`.
    kernel_size: 
        The blurring kernel size.
    border_type: 
        The padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
    separable: 
        Run as composition of two 1d-convolutions.

    Returns
    -------
    Tensor[B,C,H,W]:
        The blurred tensor with shape :math:`(B,C,H,W)`.

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 7)
    >>> output = box_blur(input, (3, 3))  # 2x4x5x7
    >>> output.shape
    torch.Size([2, 4, 5, 7])
    """
    assert_tensor(input)

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        kernel_y = get_box_kernel1d(ky, device=input.device, dtype=input.dtype)
        kernel_x = get_box_kernel1d(kx, device=input.device, dtype=input.dtype)
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
        out = filter2d(input, kernel, border_type)

    return out


class BoxBlur(nn.Module):
    r"""Blur an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Parameters
    ----------
    kernel_size: 
        The blurring kernel size.
    border_type: 
        The padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    separable: 
        Run as composition of two 1d-convolutions.

    Returns
    -------
    Tensor:
        The blurred input tensor.

    Shape
    -----
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        border_type: FilterBorder | str = "reflect",
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.separable = separable

        if separable:
            ky, kx = _unpack_2d_ks(self.kernel_size)
            self.register_buffer("kernel_y", get_box_kernel1d(ky))
            self.register_buffer("kernel_x", get_box_kernel1d(kx))
            self.kernel_y: Tensor
            self.kernel_x: Tensor
        else:
            self.register_buffer("kernel", get_box_kernel2d(kernel_size))
            self.kernel: Tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        assert_tensor(input)
        if self.separable:
            return filter2d_separable(
                input, self.kernel_x, self.kernel_y, self.border_type
            )
        return filter2d(input, self.kernel, self.border_type)


def get_motion_kernel2d(
    kernel_size: int,
    angle: Tensor | float,
    direction: Tensor | float = 0.0,
    mode: str = "nearest",
) -> Tensor:
    r"""Return 2D motion blur filter.

    Parameters
    ----------
        kernel_size: motion kernel width and height. It should be odd and positive.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Returns
    -------
        The motion blur kernel of shape :math:`(B, k_\text{size}, k_\text{size})`.

    Examples
    --------
        >>> get_motion_kernel2d(5, 0.0, 0.0)
       torch.Tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])

        >>> get_motion_kernel2d(3, 215.0, -0.5)
       torch.Tensor([[[0.0000, 0.0000, 0.1667],
                 [0.0000, 0.3333, 0.0000],
                 [0.5000, 0.0000, 0.0000]]])
    """
    device, dtype = _extract_device_dtype(
        [
            angle if isinstance(angle, Tensor) else None,
            direction if isinstance(direction, Tensor) else None,
        ]
    )

    # TODO: add support to kernel_size as tuple or integer
    kernel_tuple = _unpack_2d_ks(kernel_size)
    if not isinstance(angle, Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 0:
        angle = angle[None]

    assert_shape(angle, ("B"))

    if not isinstance(direction, Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    assert_shape(direction, ("B"))

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    # kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    # Alternatively
    # m = ((1 - 2 * direction)[:, None].repeat(1, kernel_size) / (kernel_size - 1))
    # kernel[:, kernel_size // 2, :] = direction[:, None].repeat(1, kernel_size) + m * torch.arange(0, kernel_size)
    k = stack(
        [
            (direction + ((1 - 2 * direction) / (kernel_size - 1)) * i)
            for i in range(kernel_size)
        ],
        -1,
    )
    kernel = nn.functional.pad(
        k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0]
    )

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel, angle, mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)
    return kernel


def get_motion_kernel3d(
    kernel_size: int,
    angle: Tensor | tuple[float, float, float],
    direction: Tensor | float = 0.0,
    mode: str = "nearest",
) -> Tensor:
    r"""Return 3D motion blur filter.

    Parameters
    ----------
    kernel_size: motion kernel width, height and depth. It should be odd and positive.
    angle: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
        If tensor, it must be :math:`(B, 3)`.
        If tuple, it must be (yaw, pitch, raw).
    direction: forward/backward direction of the motion blur.
        Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
        while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
        uniformly (but still angled) motion blur.
    mode: interpolation mode for rotating the kernel. ``'bilinear'`` or ``'nearest'``.

    Returns
    -------
    The motion blur kernel with shape :math:`(B, k_\text{size}, k_\text{size}, k_\text{size})`.

    Examples
    --------
        >>> get_motion_kernel3d(3, (0.0, 0.0, 0.0), 0.0)
       torch.Tensor([[[[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.3333, 0.3333, 0.3333],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])

        >>> get_motion_kernel3d(3, (90.0, 90.0, 0.0), -0.5)
       torch.Tensor([[[[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.5000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.3333, 0.0000],
                  [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0000, 0.1667, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """
    kernel_tuple = _unpack_3d_ks(kernel_size)
    if not isinstance(angle, Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 1:
        angle = angle[None]

    assert_shape(angle, ("B", "3"))

    if not isinstance(direction, Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction[None]

    assert_shape(direction, ("B"))

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0
    kernel = zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    k = stack(
        [
            (direction + ((1 - 2 * direction) / (kernel_size - 1)) * i)
            for i in range(kernel_size)
        ],
        -1,
    )
    kernel = nn.functional.pad(
        k[:, None, None],
        [
            0,
            0,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            0,
            0,
        ],
    )

    expected_shape = torch.Size([direction.size(0), *kernel_tuple])
    kernel = kernel[:, None, ...]

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate3d(
        kernel, angle[:, 0], angle[:, 1], angle[:, 2], mode=mode, align_corners=True
    )
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2, 3), keepdim=True)

    return kernel


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def _unpack_3d_ks(kernel_size: tuple[int, int, int] | int) -> tuple[int, int, int]:
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        kz, ky, kx = kernel_size

    kz = int(kz)
    ky = int(ky)
    kx = int(kx)

    return (kz, ky, kx)


def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    assert_shape(input, (..., "H", "W"))

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm[..., None, None])


def gaussian(
    window_size: int,
    sigma: Tensor | float,
    *,
    mean: Union[Tensor, float] | None = None,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Compute the gaussian values based on the window and sigma values.

    Parameters
    ----------
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        mean: Mean of the Gaussian function (center). If not provided, it defaults to window_size // 2.
        If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.
    Returns
    -------
        A tensor withshape :math:`(B, \text{kernel_size})`, with Gaussian values.
    """

    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    assert_tensor(sigma)
    assert_shape(sigma, ("B", "1"))
    batch_size = sigma.shape[0]

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, float):
        mean = torch.tensor([[mean]], device=sigma.device, dtype=sigma.dtype)

    assert_tensor(mean)
    assert_shape(mean, ("B", "1"))

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - mean
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def gaussian_discrete_erf(
    window_size: int,
    sigma: Tensor | float,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Discrete Gaussian by interpolating the error function.

    Parameters
    ----------
    window_size: the size which drives the filter amount.
    sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
    device: This value will be used if sigma is a float. Device desired to compute.
    dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(B, \text{kernel_size})`, with discrete Gaussian
        values computed by approximation of the error function.
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    assert_shape(sigma, ("B", "1"))
    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    t = 0.70710678 / sigma.abs()
    # t = torch.tensor(2, device=sigma.device, dtype=sigma.dtype).sqrt() / (sigma.abs() * 2)

    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss = gauss.clamp(min=0)

    return gauss / gauss.sum(-1, keepdim=True)


def _modified_bessel_0(x: Tensor) -> Tensor:
    ax = torch.abs(x)
    out = zeros_like(x)
    idx_a = ax < 3.75
    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        out[idx_a] = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))
            )
        )

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.916281e-2 + y * (
            -0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))
        )
        coef = 0.39894228 + y * (
            0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans))
        )
        out[idx_b] = (ax[idx_b].exp() / ax[idx_b].sqrt()) * coef

    return out


def _modified_bessel_1(x: Tensor) -> Tensor:
    ax = torch.abs(x)
    out = zeros_like(x)
    idx_a = ax < 3.75
    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        ans = 0.51498869 + y * (
            0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))
        )
        out[idx_a] = ax[idx_a] * (0.5 + y * (0.87890594 + y * ans))

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
        ans = 0.39894228 + y * (
            -0.3988024e-1
            + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans)))
        )
        ans = ans * ax[idx_b].exp() / ax[idx_b].sqrt()
        out[idx_b] = where(x[idx_b] < 0, -ans, ans)

    return out


def _modified_bessel_i(n: int, x: Tensor) -> Tensor:
    if (x == 0.0).all():
        return x

    batch_size = x.shape[0]

    tox = 2.0 / x.abs()
    ans = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bip = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bi = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

    m = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        idx = bi.abs() > 1.0e10

        if idx.any():
            ans[idx] = ans[idx] * 1.0e-10
            bi[idx] = bi[idx] * 1.0e-10
            bip[idx] = bip[idx] * 1.0e-10

        if j == n:
            ans = bip

    out = ans * _modified_bessel_0(x) / bi

    if (n % 2) == 1:
        out = where(x < 0.0, -out, out)

    # TODO: skip the previous computation for x == 0, instead of forcing here
    out = where(x == 0.0, x, out)

    return out


def gaussian_discrete(
    window_size: int,
    sigma: Tensor | float,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Discrete Gaussian kernel based on the modified Bessel functions.

    Parameters
    ----------
    window_size: the size which drives the filter amount.
    sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
    device: This value will be used if sigma is a float. Device desired to compute.
    dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
    A tensor with shape :math:`(B, \text{kernel_size})`, with discrete Gaussian values
    computed by modified Bessel function.
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    assert_shape(sigma, ("B", "1"))

    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [
        _modified_bessel_0(sigma2),
        _modified_bessel_1(sigma2),
        *(_modified_bessel_i(k, sigma2) for k in range(2, tail)),
    ]
    out = torch.cat(bessels[:0:-1] + bessels, -1) * sigma2.exp()

    return out / out.sum(-1, keepdim=True)


def laplacian_1d(
    window_size: int, *, device: Device | None = None, dtype: DType = torch.float32
) -> Tensor:
    r"""
    One could also use the Laplacian of Gaussian formula to design the filter.
    """
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(
    kernel_size: int, *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""Utility function that returns a 1-D box filter.

    Parameters
    ----------
    kernel_size: the size of the kernel.
    device: the desired device of returned tensor.
    dtype: the desired data type of returned tensor.

    Returns
    -------
    A tensor with shape :math:`(1, \text{kernel\_size})`, filled with the value
    :math:`\frac{1}{\text{kernel\_size}}`.
    """
    scale = torch.tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(1, kernel_size)


def get_box_kernel2d(
    kernel_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 2-D box filter.

    Parameters
    ----------
    kernel_size: the size of the kernel.
    device: the desired device of returned tensor.
    dtype: the desired data type of returned tensor.

    Returns
    -------
    A tensor with shape :math:`(1, \text{kernel\_size}[0], \text{kernel\_size}[1])`,
    filled with the value :math:`\frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]}`.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    scale = torch.tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    r"""
    Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def get_sobel_kernel_5x5_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _get_sobel_kernel_5x5_2nd_order_xy(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel_3x3(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor(
        [[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel3d(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_diff_kernel3d_2nd_order(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel2d(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_diff_kernel2d(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    gxx = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
    )
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    return stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode: str,
    order: int,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    if mode == "sobel" and order == 1:
        kernel: Tensor = get_sobel_kernel2d(device=device, dtype=dtype)
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented for order {order} on mode {mode}")

    return kernel


def get_spatial_gradient_kernel3d(
    mode: str,
    order: int,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid gradients, using one of the following
    operators: sobel, diff."""
    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(
            f"Not implemented 3d gradient kernel for order {order} on mode {mode}"
        )

    return kernel


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Parameters
    ----------
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
        gaussian filter coefficients with shape :math:`(B, \text{kernel_size})`.

    Examples
    --------
        >>> get_gaussian_kernel1d(3, 2.5)
       torch.Tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
       torch.Tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
       torch.Tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])
    """
    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_discrete_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Function that returns Gaussian filter coefficients based on the modified Bessel functions.

    Parameters
    ----------
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
        1D tensor with gaussian filter coefficients. With shape :math:`(B, \text{kernel_size})`

    Examples
    --------
        >>> get_gaussian_discrete_kernel1d(3, 2.5)
       torch.Tensor([[0.3235, 0.3531, 0.3235]])
        >>> get_gaussian_discrete_kernel1d(5, 1.5)
       torch.Tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096]])
        >>> get_gaussian_discrete_kernel1d(5, torch.tensor([[1.5], [2.4]]))
       torch.Tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096],
                [0.1635, 0.2170, 0.2389, 0.2170, 0.1635]])
    """
    return gaussian_discrete(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_erf_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Function that returns Gaussian filter coefficients by interpolating the error function.

    Parameters
    ----------
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
        1D tensor with gaussian filter coefficients. Shape :math:`(B, \text{kernel_size})`

    Examples
    --------
        >>> get_gaussian_erf_kernel1d(3, 2.5)
       torch.Tensor([[0.3245, 0.3511, 0.3245]])
        >>> get_gaussian_erf_kernel1d(5, 1.5)
       torch.Tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226]])
        >>> get_gaussian_erf_kernel1d(5, torch.tensor([[1.5], [2.1]]))
       torch.Tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226],
                [0.1574, 0.2198, 0.2456, 0.2198, 0.1574]])
    """
    return gaussian_discrete_erf(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Parameters
    ----------
        kernel_size: filter sizes in the y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the y and x.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
        2D tensor with gaussian filter matrix coefficients.

    Shape
    -----
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y)`

    Examples
    --------
        >>> get_gaussian_kernel2d((5, 5), (1.5, 1.5))
       torch.Tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
       torch.Tensor([[[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                 [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                 [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]]])
        >>> get_gaussian_kernel2d((5, 5), torch.tensor([[1.5, 1.5]]))
       torch.Tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    assert_tensor(sigma)
    assert_shape(sigma, ("B", "2"))

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(
        ksize_y, sigma_y, force_even, device=device, dtype=dtype
    )[..., None]
    kernel_x = get_gaussian_kernel1d(
        ksize_x, sigma_x, force_even, device=device, dtype=dtype
    )[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: tuple[int, int, int] | int,
    sigma: tuple[float, float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Parameters
    ----------
        kernel_size: filter sizes in the z, y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the z, y and x direction.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
        3D tensor with gaussian filter matrix coefficients.

    Shape
    -----
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y,  \text{kernel_size}_z)`

    Examples
    --------
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5))
       torch.Tensor([[[[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]],
        <BLANKLINE>
                 [[0.0364, 0.0455, 0.0364],
                  [0.0455, 0.0568, 0.0455],
                  [0.0364, 0.0455, 0.0364]],
        <BLANKLINE>
                 [[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]]]])
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).sum()
       torch.Tensor(1.)
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).shape
        torch.Size([1, 3, 3, 3])
        >>> get_gaussian_kernel3d((3, 7, 5), torch.tensor([[1.5, 1.5, 1.5]])).shape
        torch.Size([1, 3, 7, 5])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    assert_tensor(sigma)
    assert_shape(sigma, ("B", "3"))

    ksize_z, ksize_y, ksize_x = _unpack_3d_ks(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]

    kernel_z = get_gaussian_kernel1d(
        ksize_z, sigma_z, force_even, device=device, dtype=dtype
    )
    kernel_y = get_gaussian_kernel1d(
        ksize_y, sigma_y, force_even, device=device, dtype=dtype
    )
    kernel_x = get_gaussian_kernel1d(
        ksize_x, sigma_x, force_even, device=device, dtype=dtype
    )

    return (
        kernel_z.view(-1, ksize_z, 1, 1)
        * kernel_y.view(-1, 1, ksize_y, 1)
        * kernel_x.view(-1, 1, 1, ksize_x)
    )


def get_laplacian_kernel1d(
    kernel_size: int, *, device: Device | None = None, dtype: DType = torch.float32
) -> Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Parameters
    ----------
        kernel_size: filter size. It should be odd and positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns
    -------
        1D tensor with laplacian filter coefficients.

    Shape
    -----
        - Output: math:`(\text{kernel_size})`

    Examples
    --------
        >>> get_laplacian_kernel1d(3)
       torch.Tensor([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
       torch.Tensor([ 1.,  1., -4.,  1.,  1.])
    """
    return laplacian_1d(kernel_size, device=device, dtype=dtype)


def get_laplacian_kernel2d(
    kernel_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Parameters
    ----------
        kernel_size: filter size should be odd.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns
    -------
        2D tensor with laplacian filter matrix coefficients.

    Shape
    -----
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples
    --------
        >>> get_laplacian_kernel2d(3)
       torch.Tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
       torch.Tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x = kx // 2
    mid_y = ky // 2

    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel


def get_pascal_kernel_2d(
    kernel_size: tuple[int, int] | int,
    norm: bool = True,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
     Generate pascal filter kernel by kernel size.

     Parameters
     ----------
         kernel_size: height and width of the kernel.
         norm: if to normalize the kernel or not. Default: True.
         device: tensor device desired to create the kernel
         dtype: tensor dtype desired to create the kernel

     Returns
     -------
         if kernel_size is an integer the kernel will be shaped as :math:`(kernel_size, kernel_size)`
         otherwise the kernel will be shaped as :math: `kernel_size`

     Examples
     --------
     >>> get_pascal_kernel_2d(1)
    torch.Tensor([[1.]])
     >>> get_pascal_kernel_2d(4)
    torch.Tensor([[0.0156, 0.0469, 0.0469, 0.0156],
             [0.0469, 0.1406, 0.1406, 0.0469],
             [0.0469, 0.1406, 0.1406, 0.0469],
             [0.0156, 0.0469, 0.0469, 0.0156]])
     >>> get_pascal_kernel_2d(4, norm=False)
    torch.Tensor([[1., 3., 3., 1.],
             [3., 9., 9., 3.],
             [3., 9., 9., 3.],
             [1., 3., 3., 1.]])
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    ax = get_pascal_kernel_1d(kx, device=device, dtype=dtype)
    ay = get_pascal_kernel_1d(ky, device=device, dtype=dtype)

    filt = ay[:, None] * ax[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt


def get_pascal_kernel_1d(
    kernel_size: int,
    norm: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
     Generate Yang Hui triangle (Pascal's triangle) by a given number.

     Parameters
     ----------
         kernel_size: height and width of the kernel.
         norm: if to normalize the kernel or not. Default: False.
         device: tensor device desired to create the kernel
         dtype: tensor dtype desired to create the kernel

     Returns
     -------
         kernel shaped as :math:`(kernel_size,)`

     Examples
     --------
     >>> get_pascal_kernel_1d(1)
    torch.Tensor([1.])
     >>> get_pascal_kernel_1d(2)
    torch.Tensor([1., 1.])
     >>> get_pascal_kernel_1d(3)
    torch.Tensor([1., 2., 1.])
     >>> get_pascal_kernel_1d(4)
    torch.Tensor([1., 3., 3., 1.])
     >>> get_pascal_kernel_1d(5)
    torch.Tensor([1., 4., 6., 4., 1.])
     >>> get_pascal_kernel_1d(6)
    torch.Tensor([ 1.,  5., 10., 10.,  5.,  1.])
    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_canny_nms_kernel(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hysteresis_kernel(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hanning_kernel1d(
    kernel_size: int, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Returns Hanning (also known as Hann) kernel, used in signal processing and KCF tracker.

    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    See further in numpy docs https://numpy.org/doc/stable/reference/generated/numpy.hanning.html

    Parameters
    ----------
        kernel_size: The size the of the kernel. It should be positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns
    -------
        1D tensor with Hanning filter coefficients. Shape math:`(\text{kernel_size})`
        .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)

    Examples
    --------
        >>> get_hanning_kernel1d(4)
       torch.Tensor([0.0000, 0.7500, 0.7500, 0.0000])
    """
    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / float(kernel_size - 1))
    return x


def get_hanning_kernel2d(
    kernel_size: tuple[int, int] | int,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Returns 2d Hanning kernel, used in signal processing and KCF tracker.

    Parameters
    ----------
        kernel_size: The size of the kernel for the filter. It should be positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns
    -------
        2D tensor with Hanning filter coefficients. Shape
    ----- math:`(\text{kernel_size[0], kernel_size[1]})`
        .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
    """
    kernel_size = _unpack_2d_ks(kernel_size)
    ky = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    kernel2d = ky @ kx

    return kernel2d


def _preprocess_fast_guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    subsample: int = 1,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    ky, kx = _unpack_2d_ks(kernel_size)
    if subsample > 1:
        s = 1 / subsample
        guidance_sub = interpolate(guidance, scale_factor=s, mode="nearest")
        input_sub = (
            guidance_sub
            if input is guidance
            else interpolate(input, scale_factor=s, mode="nearest")
        )
        ky, kx = ((k - 1) // subsample + 1 for k in (ky, kx))
    else:
        guidance_sub = guidance
        input_sub = input
    return guidance_sub, input_sub, (ky, kx)


def _guided_blur_grayscale_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(
        guidance, input, kernel_size, subsample
    )

    mean_I = box_blur(guidance_sub, kernel_size, border_type)
    corr_I = box_blur(guidance_sub.square(), kernel_size, border_type)
    var_I = corr_I - mean_I.square()

    if input is guidance:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type)
        corr_Ip = box_blur(guidance_sub * input_sub, kernel_size, border_type)
        cov_Ip = corr_Ip - mean_I * mean_p

    if isinstance(eps, Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> NCHW

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)

    if subsample > 1:
        mean_a = nn.functional.interpolate(
            mean_a, scale_factor=subsample, mode="bilinear"
        )
        mean_b = nn.functional.interpolate(
            mean_b, scale_factor=subsample, mode="bilinear"
        )

    return mean_a * guidance + mean_b


def _guided_blur_multichannel_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(
        guidance, input, kernel_size, subsample
    )
    B, C, H, W = guidance_sub.shape

    mean_I = box_blur(guidance_sub, kernel_size, border_type).permute(0, 2, 3, 1)
    II = (guidance_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
    corr_I = box_blur(II, kernel_size, border_type).permute(0, 2, 3, 1)
    var_I = corr_I.reshape(B, H, W, C, C) - mean_I.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type).permute(0, 2, 3, 1)
        Ip = (input_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
        corr_Ip = box_blur(Ip, kernel_size, border_type).permute(0, 2, 3, 1)
        cov_Ip = corr_Ip.reshape(B, H, W, C, -1) - mean_p.unsqueeze(
            -2
        ) * mean_I.unsqueeze(-1)

    if isinstance(eps, Tensor):
        _eps = torch.eye(C, device=guidance.device, dtype=guidance.dtype).view(
            1, 1, 1, C, C
        ) * eps.view(-1, 1, 1, 1, 1)
    else:
        _eps = guidance.new_full((C,), eps).diag().view(1, 1, 1, C, C)
    a = torch.linalg.solve(var_I + _eps, cov_Ip)  # B, H, W, C_guidance, C_input
    b = mean_p - (mean_I.unsqueeze(-2) @ a).squeeze(-2)  # B, H, W, C_input

    mean_a = box_blur(a.flatten(-2).permute(0, 3, 1, 2), kernel_size, border_type)
    mean_b = box_blur(b.permute(0, 3, 1, 2), kernel_size, border_type)

    if subsample > 1:
        mean_a = nn.functional.interpolate(
            mean_a, scale_factor=subsample, mode="bilinear"
        )
        mean_b = nn.functional.interpolate(
            mean_b, scale_factor=subsample, mode="bilinear"
        )
    mean_a = mean_a.view(B, C, -1, H * subsample, W * subsample)

    # einsum might not be contiguous, thus mean_b is the first argument
    return mean_b + torch.einsum("BCHW,BCcHW->BcHW", guidance, mean_a)


def guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    r"""Blur a tensor using a Guided filter.

    .. image:: _static/img/guided_blur.png

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Parameters
    ----------
        guidance: the guidance tensor with shape :math:`(B,C,H,W)`.
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns
    -------
        the blurred tensor with same shape as `input` :math:`(B, C, H, W)`.

    Examples
    --------
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = guided_blur(guidance, input, 3, 0.1)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    assert_tensor(guidance)
    assert_shape(guidance, ("B", "C", "H", "W"))
    if input is not guidance:
        assert_tensor(input)
        assert_shape(input, ("B", "C", "H", "W"))

    if guidance.shape[1] == 1:
        return _guided_blur_grayscale_guidance(
            guidance, input, kernel_size, eps, border_type, subsample
        )
    return _guided_blur_multichannel_guidance(
        guidance, input, kernel_size, eps, border_type, subsample
    )


class GuidedBlur(nn.Module):
    r"""Blur a tensor using a Guided filter.

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Parameters
    ----------
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns
    -------
        the blurred input tensor.

    Shape
    -----
        - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples
    --------
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = GuidedBlur(3, 0.1)
        >>> output = blur(guidance, input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        eps: float,
        border_type: FilterBorder | str = "reflect",
        subsample: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.border_type = border_type
        self.subsample = subsample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"eps={self.eps}, "
            f"border_type={self.border_type}, "
            f"subsample={self.subsample})"
        )

    @TX.override
    def forward(self, guidance: Tensor, input: Tensor) -> Tensor:
        return guided_blur(
            guidance,
            input,
            self.kernel_size,
            self.eps,
            self.border_type,
            self.subsample,
        )


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def median_blur(input: Tensor, kernel_size: tuple[int, int] | int) -> Tensor:
    r"""Blur an image using the median filter.

    Parameters
    ----------
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns
    -------
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    Example
    -------
    >>> input = torch.rand(2, 4, 5, 7)
    >>> output = median_blur(input, (3, 3))
    >>> output.shape
    torch.Size([2, 4, 5, 7])
    """
    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: Tensor = get_binary_kernel2d(
        kernel_size, device=input.device, dtype=input.dtype
    )
    b, c, h, w = input.shape

    # map the local window to single vector
    features: Tensor = nn.functional.conv2d(
        input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    return features.median(dim=2)[0]


class MedianBlur(nn.Module):
    r"""Blur an image using the median filter.

    Parameters
    ----------
    kernel_size: the blurring kernel size.

    Returns
    -------
    the blurred input tensor.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 7)
    >>> blur = MedianBlur((3, 3))
    >>> output = blur(input)
    >>> output.shape
    torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: tuple[int, int] | int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        return median_blur(input, self.kernel_size)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    "Single implementation for both Bilateral Filter and Joint Bilateral Filter"

    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))
    if guidance is not None:
        # NOTE: allow guidance and input having different number of channels
        assert_tensor(guidance)
        assert_shape(guidance, ("B", "C", "H", "W"))

    if isinstance(sigma_color, Tensor):
        assert_shape(sigma_color, ("B"))
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(
            -1, 1, 1, 1, 1
        )

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = nn.functional.pad(
        input, (pad_x, pad_x, pad_y, pad_y), mode=border_type
    )
    unfolded_input = (
        padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
    )  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = nn.functional.pad(
            guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type
        )
        unfolded_guidance = (
            padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
        )  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (
        -0.5 / sigma_color**2 * color_distance_sq
    ).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(
        kernel_size, sigma_space, device=input.device, dtype=input.dtype
    )
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""Blur a tensor using a Bilateral filter.

    .. image:: _static/img/bilateral_blur.png

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Parameters
    ----------
    input: the input tensor with shape :math:`(B,C,H,W)`.
    kernel_size: the size of the kernel.
    sigma_color: the standard deviation for intensity/color Gaussian kernel.
        Smaller values preserve more edges.
    sigma_space: the standard deviation for spatial Gaussian kernel.
        This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: the padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    color_distance_type: the type of distance to calculate intensity/color
        difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
        match OpenCV implementation. Use ``'l2'`` to match Matlab implementation.
        Default: ``'l1'``.

    Returns
    -------
    The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 5)
    >>> output = bilateral_blur(input, (3, 3), 0.1, (1.5, 1.5))
    >>> output.shape
    torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(
        input,
        None,
        kernel_size,
        sigma_color,
        sigma_space,
        border_type,
        color_distance_type,
    )


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""Blur a tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :func:`bilateral_blur()` for more information.

    Parameters
    ----------
    input: the input tensor with shape :math:`(B,C,H,W)`.
    guidance: the guidance tensor with shape :math:`(B,C,H,W)`.
    kernel_size: the size of the kernel.
    sigma_color: the standard deviation for intensity/color Gaussian kernel.
        Smaller values preserve more edges.
    sigma_space: the standard deviation for spatial Gaussian kernel.
        This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: the padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    color_distance_type: the type of distance to calculate intensity/color
        difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
        match OpenCV implementation.

    Returns
    -------
    The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 5)
    >>> guidance = torch.rand(2, 4, 5, 5)
    >>> output = joint_bilateral_blur(input, guidance, (3, 3), 0.1, (1.5, 1.5))
    >>> output.shape
    torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(
        input,
        guidance,
        kernel_size,
        sigma_color,
        sigma_space,
        border_type,
        color_distance_type,
    )


# trick to make mypy not throw errors about difference in .forward() signatures of subclass and superclass
class _BilateralBlur(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: FilterBorder | str = "reflect",
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    r"""Blur a tensor using a Bilateral filter.

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Parameters
    ----------
    kernel_size: the size of the kernel.
    sigma_color: the standard deviation for intensity/color Gaussian kernel.
        Smaller values preserve more edges.
    sigma_space: the standard deviation for spatial Gaussian kernel.
        This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: the padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    color_distance_type: the type of distance to calculate intensity/color
        difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
        match OpenCV implementation. Use ``'l2'`` to match Matlab implementation.
        Default: ``'l1'``.

    Returns
    -------
    The blurred input tensor.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 5)
    >>> blur = BilateralBlur((3, 3), 0.1, (1.5, 1.5))
    >>> output = blur(input)
    >>> output.shape
    torch.Size([2, 4, 5, 5])
    """

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )


class JointBilateralBlur(_BilateralBlur):
    r"""Blur a tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :class:`BilateralBlur` for more information.

    Parameters
    ----------
    kernel_size: the size of the kernel.
    sigma_color: the standard deviation for intensity/color Gaussian kernel.
        Smaller values preserve more edges.
    sigma_space: the standard deviation for spatial Gaussian kernel.
        This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: the padding mode to be applied before convolving.
        The expected modes are: ``'constant'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    color_distance_type: the type of distance to calculate intensity/color
        difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
        match OpenCV implementation.

    Returns
    -------
    The blurred input tensor.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> input = torch.rand(2, 4, 5, 5)
    >>> guidance = torch.rand(2, 4, 5, 5)
    >>> blur = JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))
    >>> output = blur(input, guidance)
    >>> output.shape
    torch.Size([2, 4, 5, 5])
    """

    @TX.override
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )
