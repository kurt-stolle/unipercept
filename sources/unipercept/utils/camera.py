from __future__ import annotations

import typing as T

import torch
from torch import Size, Tensor, nn


def build_calibration_matrix(
    focal_lengths: T.List[T.Tuple[float, float]] | Tensor,
    principal_points: T.List[T.Tuple[float, float]] | Tensor,
    orthographic: bool,
) -> Tensor:
    r"""
    Build the calibration matrix (K-matrix) using the focal lengths and
    principal points.
    """

    # Focal length, shape (), (N, 1) or (N, 2)
    if not isinstance(focal_lengths, Tensor):
        focal_lengths = torch.tensor([list(t) for t in focal_lengths])
    if focal_lengths.ndim in [0, 1] or focal_lengths.shape[1] == 1:
        fx = fy = focal_lengths
    else:
        fx, fy = focal_lengths.unbind(1)

    # Principal point
    if not isinstance(principal_points, Tensor):
        principal_points = torch.tensor([list(p) for p in principal_points])
    px, py = principal_points.unbind(1)

    # Amount of cameras
    N = len(focal_lengths)

    # Value checks
    assert N == len(principal_points)
    assert focal_lengths.device == principal_points.device

    # Create calibration matrix
    K: Tensor = fx.new_zeros(N, 4, 4)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    if orthographic:
        K[:, 0, 3] = px
        K[:, 1, 3] = py
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
    else:
        K[:, 0, 2] = px
        K[:, 1, 2] = py
        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0

    return K


def create_intrinsic_maps(
    shape: Size | T.Sequence[int], intrinsics: Tensor | T.Sequence[float]
) -> Tensor:
    r"""
    Encode the camera intrinsic parameters (focal length and principle point) to a map
    with channels for the distance to the principle point and the field of view.
    """
    *_, height, width = shape
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = torch.arange(0, width, dtype=torch.float32)
    x_row_center_norm = (x_row - u0) / width
    x_center = torch.tile(x_row_center_norm, (height, 1))  # [H, W]

    y_col = torch.arange(0, height, dtype=torch.float32)
    y_col_center_norm = (y_col - v0) / height
    y_center = torch.tile(y_col_center_norm, (width, 1)).T

    # FoV
    fov_x = torch.arctan(x_center / (f / width))
    fov_y = torch.arctan(y_center / (f / height))

    cam_model = torch.stack([x_center, y_center, fov_x, fov_y], dim=2)
    return cam_model


def generate_rays(
    camera_intrinsics: Tensor, image_shape: T.Tuple[int, int], noisy: bool = False
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = nn.functional.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


def spherical_zbuffer_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def spherical_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    r = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical(spherical_tensor: Tensor) -> Tensor:
    x = spherical_tensor[..., 0]  # Extract polar angle
    y = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(x / r, z / r)
    phi = torch.acos(y / r)

    euclidean_tensor = torch.stack((theta, phi, r), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical_zbuffer(euclidean_tensor: Tensor) -> Tensor:
    pitch = torch.asin(euclidean_tensor[..., 1])
    yaw = torch.atan2(euclidean_tensor[..., 0], euclidean_tensor[..., -1])
    z = euclidean_tensor[..., 2]  # Extract zbuffer depth
    euclidean_tensor = torch.stack((pitch, yaw, z), dim=-1)
    return euclidean_tensor


def unproject_points(depth: Tensor, camera_intrinsics: Tensor) -> Tensor:
    r"""
    Unprojects a batch of depth maps to 3D point clouds using camera intrinsics.

    Parameters
    ----------
    depth: Tensor[B, 1, H, W]
        Batch of depth maps.
    camera_intrinsics: Tensor[B, 3, 3]
        Camera intrinsic matrix of shape.

    Returns
    -------
    Tensor[B, 3, H, W]
        Batch of 3D point clouds.
    """
    batch_size, _, height, width = depth.shape
    device = depth.device

    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixel_coords = torch.stack((x_coords, y_coords), dim=-1)  # (H, W, 2)

    # Get homogeneous coords (u v 1)
    pixel_coords_homogeneous = torch.cat(
        (pixel_coords, torch.ones((height, width, 1), device=device)), dim=-1
    )
    pixel_coords_homogeneous = pixel_coords_homogeneous.permute(2, 0, 1).flatten(
        1
    )  # (3, H*W)
    # Apply K^-1 @ (u v 1): [B, 3, 3] @ [3, H*W] -> [B, 3, H*W]
    unprojected_points = torch.matmul(
        torch.inverse(camera_intrinsics), pixel_coords_homogeneous
    )  # (B, 3, H*W)
    unprojected_points = unprojected_points.view(
        batch_size, 3, height, width
    )  # (B, 3, H, W)
    unprojected_points = unprojected_points * depth  # (B, 3, H, W)
    return unprojected_points


def project_points(
    points_3d: Tensor,
    intrinsic_matrix: Tensor,
    image_shape: T.Tuple[int, int],
) -> Tensor:
    r"""
    Project 3D points onto the image plane using camera intrinsics:

    :math:`(u v w) = (x y z) @ K^T`

    Parameters
    ----------
    points_3d: Tensor[B, N, 3]
        Batch of 3D points.
    intrinsic_matrix: Tensor[B, 3, 3]
        Camera intrinsic matrix.
    image_shape: Tuple[int, int]
        Image shape (height, width).

    Returns
    -------
    Tensor[B, N, 2]
        Batch of 2D points.
    """
    points_2d = torch.matmul(points_3d, intrinsic_matrix.transpose(1, 2))

    # Normalize projected points: (u v w) -> (u / w, v / w, 1)
    points_2d = points_2d[..., :2] / points_2d[..., 2:]

    points_2d = points_2d.int()

    # points need to be inside the image (can it diverge onto all points out???)
    valid_mask = (
        (points_2d[..., 0] >= 0)
        & (points_2d[..., 0] < image_shape[1])
        & (points_2d[..., 1] >= 0)
        & (points_2d[..., 1] < image_shape[0])
    )

    # Calculate the flat indices of the valid pixels
    flat_points_2d = points_2d[..., 0] + points_2d[..., 1] * image_shape[1]
    flat_indices = flat_points_2d.long()

    # Create depth maps and counts using scatter_add, (B, H, W)
    depth_maps = torch.zeros(
        [points_3d.shape[0], *image_shape], device=points_3d.device
    )
    counts = torch.zeros([points_3d.shape[0], *image_shape], device=points_3d.device)

    # Loop over batches to apply masks and accumulate depth/count values
    for i in range(points_3d.shape[0]):
        valid_indices = flat_indices[i, valid_mask[i]]
        depth_maps[i].view(-1).scatter_add_(
            0, valid_indices, points_3d[i, valid_mask[i], 2]
        )
        counts[i].view(-1).scatter_add_(
            0, valid_indices, torch.ones_like(points_3d[i, valid_mask[i], 2])
        )

    # Calculate mean depth for each pixel in each batch
    mean_depth_maps = depth_maps / counts.clamp(min=1.0)
    return mean_depth_maps.reshape(-1, 1, *image_shape)  # (B, 1, H, W)


def downsample(data: Tensor, factor: int):
    """
    Downsample the input data tensor by taking the minimum value of each
    factor x factor block.

    Parameters
    ----------
    data: Tensor[B, C, H, W]
        Input data tensor.
    factor: int
        Downsampling factor.

    Returns
    -------
    Tensor[B, C, H // factor, W // factor]
        Downsampled data tensor.
    """
    N, _, H, W = data.shape
    data = data.view(
        N,
        H // factor,
        factor,
        W // factor,
        factor,
        1,
    )
    data = data.permute(0, 1, 3, 5, 2, 4).contiguous()
    data = data.view(-1, factor * factor)
    data_tmp = torch.where(data <= 0, torch.inf, data)
    data = torch.min(data_tmp, dim=-1).values
    data = data.view(N, 1, H // factor, W // factor)
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data


def flat_interpolate(
    flat_tensor: Tensor,
    shape_cur: T.Tuple[int, int],
    shape_new: T.Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> Tensor:
    """
    Interpolates a flat tensor of shape (B, C, H * W) to a new shape (B, C, H * W).

    Parameters
    ----------
    flat_tensor: Tensor[B, C, H * W]
        Input flat tensor.
    shape_cur: Tuple[int, int]
        Current shape of the tensor.
    shape_new: Tuple[int, int]
        New shape of the tensor.
    antialias: bool
        Whether to use antialiasing.
    mode: str
        Interpolation mode.

    Returns
    -------
    Tensor[B, C, H * W]
        Interpolated flat tensor.
    """

    if shape_cur == shape_new:
        return flat_tensor

    tensor = flat_tensor.view(
        flat_tensor.shape[0], shape_cur[0], shape_cur[1], -1
    ).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = nn.functional.interpolate(
        tensor,
        size=(shape_new[0], shape_new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, shape_new[0] * shape_new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()
