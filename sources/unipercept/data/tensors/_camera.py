r"""
Implements the camera s a TVTensor.
"""

from __future__ import annotations

import typing as T
import torch
from torch import Tensor, Size

from torchvision.transforms.v2.functional import register_kernel
from torchvision.tv_tensors import TVTensor

from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from unipercept.utils.camera import build_extrinsic_matrix, build_intrinsic_matrix

__all__ = ["PinholeCamera"]


class PinholeCamera(TVTensor):
    """
    A PinholeCamera is a TVTensor that represents a pinhole camera model.
    This tensor always has a shape of [..., 4, 9], where the last dimension
    represents a concatenation of the intrinsic and extrinsic parameters of the camera,
    plus an additional column that represents the image size:

    - [..., :, 0:4] represents the intrinsic matrix
    - [..., :, 4:8] represents the extrinsic matrix
    - [..., :, 8:9] represents the canvas as (u1, v1, u2, v2)
    """

    @classmethod
    def from_parameters(
        cls,
        focal_length: Tensor | T.Iterable[int | float],
        principal_point: Tensor | T.Iterable[float | int],
        rotation: Tensor | T.Iterable[float | int] | None = None,  # pitch, yaw, roll
        translation: Tensor | T.Iterable[float | int] | None = None,  # x, y, z
        canvas: Tensor | T.Iterable[float | int] | None = None,
    ) -> T.Self:
        """
        Constructs a PinholeCamera tensor from the camera parameters.
        """
        K = build_intrinsic_matrix(focal_length, principal_point, orthographic=False)
        if rotation is None and translation is None:
            E = torch.zeros_like(K)
        else:
            if rotation is None:
                translation = torch.as_tensor(translation)
                rotation = torch.zeros_like(translation)
            else:
                rotation = torch.as_tensor(rotation)
                translation = torch.as_tensor(translation)
            E = build_extrinsic_matrix(rotation, translation)

        return cls.from_parts(K, E, canvas)

    @classmethod
    def from_parts(
        cls,
        intrinsics: Tensor,
        extrinsics: Tensor | None = None,
        canvas: Tensor | None = None,
    ) -> T.Self:
        """
        Constructs a PinholeCamera tensor from the intrinsic and extrinsic matrices
        and the image size.
        """
        intrinsics = torch.as_tensor(intrinsics)
        assert intrinsics.ndim >= 2 and intrinsics.size(-1) == 4

        if extrinsics is None:
            extrinsics = torch.zeros_like(intrinsics)
        else:
            extrinsics = torch.as_tensor(extrinsics)
        assert extrinsics.ndim >= 2 and extrinsics.size(-1) == 4

        if canvas is None:
            u0, v0 = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
            canvas = torch.cat([v0 * 2, u0 * 2], dim=-1)
        else:
            canvas = torch.as_tensor(canvas).to(intrinsics)
            if canvas.size(-1) == 2:
                canvas = canvas.flip(-1)  # H, W -> W, H
                canvas = torch.cat([torch.zeros_like(canvas), canvas], dim=-1)  # bbox
            elif canvas.size(-1) != 4:
                msg = (
                    "Canvas must be a tensor of shape (V, U) or (u1, v1, u2, v2). "
                    f"Got: {canvas.shape}"
                )
                raise ValueError(msg)

        assert canvas.ndim == intrinsics.ndim - 1

        res = torch.zeros((*intrinsics.shape[:-2], 4, 10), dtype=intrinsics.dtype)
        res[..., :, 0:4] = intrinsics
        res[..., :, 4:8] = extrinsics
        res[..., :, 8] = canvas
        return res.as_subclass(cls)

    def __new__(
        cls,
        data: T.Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> T.Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        assert cls.check_pinhole_camera_tensor_shape(
            tensor
        ), f"Expected a tensor of shape [..., 2, 4, 4], but got {tensor.shape}"
        return tensor.as_subclass(cls)

    @staticmethod
    def check_pinhole_camera_tensor_shape(tensor: Tensor) -> bool:
        """
        Returns whether the input tensor has the shape of a PinholeCamera tensor.

        This is not a sufficient condition for assessing whether the tensor is a
        PinholeCamera. It only checks the shape of the tensor and whether this would
        be a valid shape for a PinholeCamera tensor.
        """
        return tensor.size(-2) == 4 and tensor.size(-1) == 9

    ########################
    # Tensor destructuring #
    ########################

    @property
    def intrinsic_matrix(self) -> Tensor:
        """
        Returns the intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 0:4]

    @property
    def extrinsic_matrix(self) -> Tensor:
        """
        Returns the extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 4:8]

    @property
    def canvas_size(self) -> Tensor:
        """
        Returns the image size as a (batch_size, (V, U)) tensor.
        """
        bbox = self.canvas_bbox
        size = bbox[..., 2:] - bbox[..., :2]
        return size.flip(-1)

    @property
    def canvas_height(self) -> Tensor:
        """
        Returns the image height as a (batch_size, V) tensor.
        """
        return self.canvas_bbox[..., 3] - self.canvas_bbox[..., 1]

    @property
    def canvas_width(self) -> Tensor:
        """
        Returns the image width as a (batch_size, U) tensor.
        """
        return self.canvas_bbox[..., 2] - self.canvas_bbox[..., 0]

    @property
    def canvas_center(self) -> Tensor:
        """
        Returns the image center as a (batch_size, (V, U)) tensor.
        """
        bbox = self.canvas_bbox
        size = self.canvas_size
        return bbox[..., :2].flip(-1) + size / 2

    @property
    def canvas_bbox(self) -> Tensor:
        """
        Returns the image crop bounding box as a [B, (u1, v1, u2, v2)] tensor.
        """
        return self[..., 8]

    ##########################
    # Sub-matrices and views #
    ##########################

    @property
    def focal_length(self) -> Tensor:
        """
        Returns the focal length of the camera as a [..., (fx, fy)] tensor.
        """
        return self[..., :2, :2].diagonal()

    @property
    def principal_point(self) -> Tensor:
        """
        Returns the principal point of the camera as a [..., (u0, v0)] tensor.
        """
        return self[..., :2, 2]

    @property
    def camera_matrix(self) -> Tensor:
        """
        Returns the camera matrix of the camera.
        """
        return self[..., :3, 0:3]

    @property
    def rotation_matrix(self) -> Tensor:
        """
        Returns the rotation part of the extrinsic parameters as a (batch_size, 3, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def translation_vector(self) -> Tensor:
        """
        Returns the translation part of the extrinsic parameters as a (batch_size, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def rotation_translation_matrix(self) -> Tensor:
        """
        Returns the concatenated rotation-translation matrix of the camera as a
        (batch_size, 3, 4) tensor.
        """
        return self[..., :3, 4:8]

    #####################
    # Modifying methods #
    #####################

    def scale_(self, factor: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`scale`.
        """
        if factor.size(-1) == 1:
            factor = factor.expand(*factor.shape[:-1], 2)
        elif factor.size(-1) != 2:
            msg = f"Expected a tensor of shape [..., (V, U)], but got {factor.shape}"
            raise ValueError(msg)

        factor = factor.to(self)
        scale_h, scale_w = factor[..., 0], factor[..., 1]

        # Modify the intrinsic matrix
        self.intrinsic_matrix[..., 0, 0] *= scale_w
        self.intrinsic_matrix[..., 1, 1] *= scale_h
        self.intrinsic_matrix[..., 0, 2] *= scale_w
        self.intrinsic_matrix[..., 1, 2] *= scale_h

        # Modify the crop (and convert H, W -> V, U)
        crop_center = (self.canvas_center * factor).flip(-1)
        crop_size = (self.canvas_size * factor).flip(-1)

        self[..., 8] = torch.cat(
            [crop_center - crop_size / 2, crop_center + crop_size / 2], dim=-1
        )

        return self

    def scale(self, factor: Tensor, inplace: bool = False) -> T.Self:
        """
        Scales the camera by the given factor.

        Assumes the scaling is done w.r.t. the center of the current image crop.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.scale_(factor)

    def crop_(self, crop: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`crop`.
        """
        self[..., 8] = crop.to(self)
        return self

    def crop(self, crop: Tensor, inplace: bool = False) -> T.Self:
        """
        Applies a crop to the camera.

        Parameters
        ----------
        crop: Tensor[..., (u1, v1, u2, v2)]
            The crop to apply to the camera.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.crop_(crop)


@register_kernel(functional="resize", tv_tensor_cls=PinholeCamera)
def resize_camera(
    camera: PinholeCamera,
    size: T.List[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = False,  # noqa: U100
    use_rescale: bool = False,
) -> Tensor:
    h_old, w_old = camera.canvas_size.tolist()
    h_new, w_new = _compute_resized_output_size(
        (h_old, w_old), size=size, max_size=max_size
    )

    return camera.scale(torch.tensor([h_new / h_old, w_new / w_old]))


@register_kernel(functional="crop", tv_tensor_cls=PinholeCamera)
def crop_camera(
    camera: PinholeCamera,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tensor:
    crop = torch.tensor([left, top, left + width, top + height])
    return camera.crop(crop)
