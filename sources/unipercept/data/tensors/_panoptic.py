from __future__ import annotations

import typing as T
from enum import StrEnum, auto
from typing import override

import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from torch import Tensor
from torchvision.tv_tensors import Mask as _Mask

from unipercept import file_io
from unipercept.data.tensors.helpers import write_png_l16, write_png_rgb
from unipercept.data.types.coco import COCOResultPanopticSegment
from unipercept.types import Pathable

from .registry import pixel_maps

__all__ = [
    "PanopticMap",
    "LabelsFormat",
    "PanopticMapLike",
    "save_panoptic",
    "read_panoptic",
]

if T.TYPE_CHECKING:
    pass

_L = T.TypeVar("_L", int, Tensor)
_BYTE_OFFSET: T.Final[T.Literal[256]] = 256


class LabelsFormat(StrEnum):
    """
    Enumerates the different formats of labels that are supported. Uses the name of
    the dataset that introduced the format.
    """

    CITYSCAPES = auto()
    CITYSCAPES_VPS = auto()
    CITYSCAPES_DVPS = auto()
    KITTI = auto()
    VISTAS = auto()
    WILD_DASH = auto()
    TORCH = auto()
    SAFETENSORS = auto()
    PNG_UINT16 = auto()
    PNG_L16 = auto()


@pixel_maps.register
class PanopticMap(_Mask):
    """
    Implements a panoptic segmentation map, where each pixel has the value:
        category_id * divisor + instance_id.
    """

    DIVISOR: T.ClassVar[int] = int(2**15)  # same for all datasets
    IGNORE: T.ClassVar[int] = -1

    @classmethod
    @torch.no_grad()
    def read(cls, path: Pathable, **meta_kwds) -> T.Self:
        return read_panoptic(path, **meta_kwds).as_subclass(cls)

    def save(
        self,
        path: Pathable,
        *,
        format: LabelsFormat | str | None = None,
        **meta_kwds,
    ) -> None:
        return save_panoptic(self, path, format=format, **meta_kwds)

    @classmethod
    def default(cls, shape: torch.Size, device: torch.device | str = "cpu") -> T.Self:
        return torch.full(
            shape, cls.IGNORE * cls.DIVISOR, dtype=torch.long, device=device
        ).as_subclass(cls)

    @classmethod
    def default_like(cls, other: Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(
            torch.full_like(
                other, fill_value=cls.IGNORE * cls.DIVISOR, dtype=torch.long
            )
        )

    @classmethod
    @override
    def wrap_like(
        cls,
        other: T.Self,
        tensor: Tensor,
    ) -> T.Self:
        return tensor.to(dtype=torch.long, non_blocking=True).as_subclass(cls)

    @classmethod
    def from_semantic(cls, semantic: Tensor) -> PanopticMap:
        """
        Create an instance from a semantic segmentation map by setting the instance IDs to 0.
        """
        if not torch.compiler.is_compiling():
            cls.must_be_semantic_map(semantic)
        return (semantic * cls.DIVISOR).as_subclass(PanopticMap)

    @classmethod
    def from_parts(cls, semantic: Tensor, instance: Tensor) -> PanopticMap:
        """
        Create an instance from a semantic segmentation and instance segmentation map by combining them
        using the global ``divisor``.
        """
        if not torch.compiler.is_compiling():
            if semantic.shape != instance.shape:
                msg = f"Expected tensors of the same shape, got {semantic.shape} and {instance.shape}"
                raise ValueError(msg)
            cls.must_be_semantic_map(semantic)
            cls.must_be_instance_map(instance)

        semantic = semantic.to(dtype=torch.long, non_blocking=True)
        instance = instance.to(dtype=torch.long, non_blocking=True)

        ignore_mask = semantic == cls.IGNORE
        panoptic = instance + semantic * cls.DIVISOR
        panoptic[ignore_mask] = cls.IGNORE

        return panoptic.as_subclass(PanopticMap)

    @classmethod
    def from_combined(cls, encoded_map: Tensor | T.Any, divisor: int) -> PanopticMap:
        """
        Decompose an encoded map into a semantic segmentation and instance segmentation map, then combine
        again using the global ``divisor``.
        """
        encoded_map = torch.as_tensor(encoded_map)
        assert encoded_map.dtype in (torch.int32, torch.int64), encoded_map.dtype

        sem_id = torch.floor_divide(encoded_map, divisor)
        ins_id = torch.remainder(encoded_map, divisor)
        ins_id = torch.where(encoded_map >= 0, ins_id, 0)

        return PanopticMap.from_parts(sem_id, ins_id)

    @T.overload
    def to_parts(self: Tensor) -> Tensor: ...

    @T.overload
    def to_parts(self: Tensor, as_tuple=True) -> tuple[Tensor, Tensor]: ...

    def to_parts(
        self: Tensor, as_tuple: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Split the semantic and instance segmentation maps, returing a tensor of size [..., 2].
        The first channel contains the semantic segmentation map, the second channel contains the instance
        id that is NOT UNIQUE for each class.
        """
        ignore_mask = self == PanopticMap.IGNORE
        sem = torch.floor_divide(self, PanopticMap.DIVISOR)
        ins = torch.remainder(self, PanopticMap.DIVISOR)
        ins[ignore_mask] = 0
        if as_tuple:
            return sem.as_subclass(_Mask), ins.as_subclass(_Mask)
        return torch.stack((sem, ins), dim=-1).as_subclass(_Mask)

    @staticmethod
    def parse_label(label: _L) -> tuple[_L, _L]:
        """
        Parse a label into a semantic and instance ID.
        """
        sem_id: _L
        ins_id: _L

        if isinstance(label, int):
            sem_id = label // PanopticMap.DIVISOR
            ins_id = label % PanopticMap.DIVISOR if label >= 0 else 0
        else:
            sem_id = torch.floor_divide(label, PanopticMap.DIVISOR)
            ins_id = torch.remainder(label, PanopticMap.DIVISOR)
            ins_id = torch.where(label >= 0, ins_id, 0)

        return sem_id, ins_id

    @classmethod
    @T.overload
    def is_void(cls, label: int) -> bool: ...

    @classmethod
    @T.overload
    def is_void(cls, label: Tensor) -> Tensor: ...

    @classmethod
    def is_void(cls, label: Tensor | int) -> Tensor | bool:
        return label < 0

    def get_semantic_map(self: Tensor) -> _Mask:
        return torch.floor_divide(self, PanopticMap.DIVISOR).as_subclass(_Mask)

    def get_semantic_masks(self: Tensor) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each semantic class."""
        sem_map = PanopticMap.get_semantic_map(self)
        uq = torch.unique(sem_map)
        yield from (
            (int(u), (sem_map == u).as_subclass(_Mask))
            for u in uq
            if u != PanopticMap.IGNORE
        )

    def get_semantic_mask(self, class_id: int) -> _Mask:
        """Return a mask for the specified semantic class."""
        return (self.get_semantic_map() == class_id).as_subclass(_Mask)

    def unique_semantics(self) -> Tensor:
        """Count the number of unique semantic classes."""
        uq = torch.unique(self.get_semantic_map())
        return uq[uq >= 0]

    def get_instance_map(self: Tensor, return_labels: bool = True) -> _Mask:
        # old: does not support same sub-id for different classes
        ins_ids = torch.remainder(self, PanopticMap.DIVISOR)
        if return_labels:
            mask = torch.where((ins_ids > 0) & (self != PanopticMap.IGNORE), self, 0)
        else:
            mask = torch.where(self != PanopticMap.IGNORE, ins_ids, 0)
        return mask.as_subclass(_Mask)

    def get_instance_masks(self: Tensor) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each instance."""
        ins_map = PanopticMap.get_instance_map(self)
        uq = torch.unique(ins_map)
        yield from ((int(u), (ins_map == u).as_subclass(_Mask)) for u in uq if u > 0)

    def get_instance_mask(self: Tensor, instance_id: int) -> _Mask:
        """Return a mask for the specified instance."""
        return (PanopticMap.get_instance_map(self) == instance_id).as_subclass(_Mask)

    def get_masks_by_label(
        self: Tensor, *, with_void: bool = False, as_tensor: bool = False
    ) -> T.Iterator[tuple[Tensor, _Mask]]:
        """
        Iterate pairs of labels and masks, where each masks corresponds to a unique
        label.
        """
        for pan_id in self.unique():
            if not as_tensor:
                pan_id = pan_id.detach().item()
            if PanopticMap.is_void(pan_id) and not with_void:
                continue
            yield pan_id, (self == pan_id).as_subclass(_Mask)

    def get_masks(self: Tensor, **kwargs) -> T.Iterator[tuple[Tensor, Tensor, _Mask]]:
        """Return a mask for each semantic class and instance (if any)."""
        for pan_id, mask in PanopticMap.get_masks_by_label(self, **kwargs):
            sem_id, ins_id = PanopticMap.parse_label(pan_id)
            yield sem_id, ins_id, mask

    def unique_instances(self: Tensor) -> Tensor:
        """Count the number of unique instances for each semantic class."""
        ins_mask = PanopticMap.get_instance_map(self) != PanopticMap.IGNORE
        return torch.unique(self[ins_mask])

    def remove_instances_(self: Tensor, semantic_list: T.Iterable[int]) -> None:
        """Remove instances for the specified semantic classes."""
        sem_map, ins_map = PanopticMap.to_parts(self, as_tuple=True)

        # Compute candidate map where all pixels that are not in the semantic list are set to -1
        can_map = torch.where(ins_map > 0, sem_map, PanopticMap.IGNORE)

        # Set all pixels that are not in the semantic list to 0
        for class_ in semantic_list:
            self[can_map == class_] = class_ * PanopticMap.DIVISOR

    def translate_semantic_(
        self: Tensor,
        translation: T.Mapping[int, int],
        inverse: bool = False,
        *,
        raises: bool = False,
    ) -> None:
        """
        Apply a translation to the class labels. The translation is a dictionary mapping old class IDs to
        new class IDs. All old class IDs that are not in the dictionary are mapped to ``void_id``.
        """
        sem_map, ins_map = PanopticMap.to_parts(self, as_tuple=True)
        self.fill_(PanopticMap.IGNORE)

        if raises:
            cur_ids = torch.unique(sem_map).tolist()
            if not all(old_id in translation for old_id in cur_ids):
                msg = f"Missing entries for given classes ({cur_ids}) in translations ({translation.keys()})"
                raise ValueError(msg)

        for (
            old_id,
            new_id,
        ) in translation.items():
            if inverse:
                old_id, new_id = new_id, old_id

            mask = sem_map == old_id
            self[mask] = new_id * PanopticMap.DIVISOR + ins_map[mask]

    def get_nonempty(self: Tensor) -> _Mask:
        """Return a new instance with only the non-empty pixels."""
        return self[self >= 0].as_subclass(_Mask)

    def to_coco(self) -> tuple[pil_image.Image, list[COCOResultPanopticSegment]]:
        segm = torch.zeros_like(self, dtype=torch.int32)

        segments_info = []

        for i, (sem_id, ins_id, mask) in enumerate(self.get_masks()):
            coco_id = i + 1
            segm[mask] = coco_id
            segments_info.append(
                COCOResultPanopticSegment(id=coco_id, category_id=sem_id)
            )

        img = pil_image.fromarray(segm.numpy().astype("uint8"), mode="L")

        return img, segments_info

    @classmethod
    def must_be_semantic_map(cls, t: Tensor):
        if t.ndim < 2:
            msg = f"Expected 2D tensor, got {t.ndim}D tensor"
        elif t.dtype not in (torch.int32, torch.int64):
            msg = f"Expected int32 or int64 tensor, got {t.dtype}"
        elif (t_min := t.min()) < cls.IGNORE:
            msg = f"Expected non-negative values other than {cls.IGNORE}, got {t_min.item()}"
        elif (t_max := t.max()) >= cls.DIVISOR:
            msg = f"Expected values < {cls.DIVISOR}, got {t_max.item()}"
        else:
            return
        raise ValueError(msg)

    @classmethod
    def is_semantic_map(cls, t: Tensor) -> bool:
        try:
            cls.must_be_semantic_map(t)
        except ValueError:
            return False
        return True

    @classmethod
    def must_be_instance_map(cls, t: Tensor):
        if t.ndim < 2:
            msg = f"Expected 2D tensor, got {t.ndim}D tensor"
        elif t.dtype not in (torch.int32, torch.int64):
            msg = f"Expected int32 or int64 tensor, got {t.dtype}"
        elif (t_min := t.min()) < cls.IGNORE:
            msg = f"Expected non-negative values, got {t_min.item()}"
        elif (t_max := t.max()) >= cls.DIVISOR:
            msg = f"Expected values < {cls.DIVISOR}, got {t_max.item()}"
        else:
            return
        raise ValueError(msg)

    @classmethod
    def is_instance_map(cls, t: Tensor) -> bool:
        try:
            cls.must_be_instance_map(t)
        except ValueError:
            return False
        return True


PanopticMapLike: T.TypeAlias = PanopticMap | Tensor


def read_panoptic(path: Pathable, /, **meta_kwds) -> PanopticMap:
    """Read a panoptic map from a file."""
    from .helpers import get_kwd, read_pixels

    data_format = get_kwd(meta_kwds, "format", LabelsFormat)
    path = str(file_io.Path(path))

    match data_format:
        case LabelsFormat.SAFETENSORS:
            labels = safetensors.load_file(path)["data"].as_subclass(PanopticMap)
        case LabelsFormat.TORCH:
            labels = torch.load(path, map_location=torch.device("cpu")).as_subclass(
                PanopticMap
            )
            assert labels is not None
            assert isinstance(labels, (PanopticMap, Tensor)), type(labels)
        case LabelsFormat.CITYSCAPES:
            divisor = 1000
            void_id = 255
            img = read_pixels(path, color=True)
            assert img.ndim == 3, f"Expected 3D tensor, got {img.ndim}D tensor"

            map_ = (
                img[:, :, 0]
                + _BYTE_OFFSET * img[:, :, 1]
                + _BYTE_OFFSET * _BYTE_OFFSET * img[:, :, 2]
            )
            map_ = torch.where(map_ > 0, map_, void_id)
            map_ = torch.where(map_ < divisor, map_ * divisor, map_ + 1)

            labels = PanopticMap.from_combined(map_, divisor)
        case LabelsFormat.CITYSCAPES_VPS:
            divisor = 1000
            void_id = 255

            img = read_pixels(path, color=False)
            assert img.ndim == 2, f"Expected 2D tensor, got {img.ndim}D tensor"

            has_instance = img >= divisor

            ids = torch.where(has_instance, (img % divisor) + 1, 0)
            sem = torch.where(has_instance, img // divisor, img)
            sem[sem == void_id] = -1

            labels = PanopticMap.from_parts(sem, ids)
        case LabelsFormat.CITYSCAPES_DVPS:
            divisor = 1000
            void_id = 32

            img = read_pixels(path, color=False)
            assert img.ndim == 2, f"Expected 2D tensor, got {img.ndim}D tensor"

            has_instance = img >= divisor

            ids = (img % divisor) + 1
            sem = img // divisor
            sem[sem == void_id] = -1

            labels = PanopticMap.from_parts(sem, ids)
        case LabelsFormat.KITTI:
            img = read_pixels(path, color=True)
            sem = img[:, :, 0]  # R-channel
            ids = torch.add(
                img[:, :, 1] * _BYTE_OFFSET,  # G channel
                img[:, :, 2],  # B channel
            )

            labels = PanopticMap.from_parts(sem, ids)
        case LabelsFormat.VISTAS:
            divisor = 1000

            img = read_pixels(path, color=False)
            assert img.dtype == torch.int32, img.dtype

            if img.ndim == 3:
                assert img.shape[2] == 3, f"Expected 3 channels, got {img.shape[2]}"
                assert torch.all(
                    img[:, :, 0] == img[:, :, 1]
                ), "Expected all channels to be equal"
                assert torch.all(
                    img[:, :, 0] == img[:, :, 2]
                ), "Expected all channels to be equal"
                img = img[:, :, 0]

            labels = PanopticMap.from_combined(img, divisor)
        case LabelsFormat.WILD_DASH:
            annotations = get_kwd(meta_kwds, "annotations", list[dict[str, T.Any]])

            divisor = int(1e8)
            void_id = 255

            img = read_pixels(path, color=True)
            img = (
                img[:, :, 0].to(torch.long) * _BYTE_OFFSET * _BYTE_OFFSET
                + img[:, :, 1].to(torch.long) * _BYTE_OFFSET
                + img[:, :, 2].to(torch.long)
            )
            sem = torch.full_like(img, void_id, dtype=torch.long)
            for ann in annotations:
                id = ann["id"]
                category_id = ann["category_id"]
                mask = img == id
                sem[mask] = category_id

            ids = torch.full_like(img, 0, dtype=torch.long)  # TODO

            labels = PanopticMap.from_parts(sem, ids)
            labels.translate_semantic_(
                translation=translations,
            )
        case _:
            msg = f"Could not read labels from {path!r} ({data_format=})"
            raise NotImplementedError(msg)

    assert labels.ndim == 2, f"Expected 2D tensor, got {labels.ndim}D tensor"

    assert (
        labels is not None
    ), f"No labels were read from '{path}' (format: {data_format})"

    if len(meta_kwds) > 0:
        raise TypeError(f"Unexpected keyword arguments: {tuple(meta_kwds.keys())}")

    assert labels.ndim == 2, f"Expected 2D tensor, got {labels.ndim}D tensor"
    return labels


def save_panoptic(
    self: Tensor,
    path: Pathable,
    *,
    format: LabelsFormat | str | None = None,
    **kwargs,
) -> None:
    """
    Save the panoptic map to a file.
    """
    if self.ndim != 2:
        msg = f"Expected 2D tensor, got {self.ndim}D tensor ({self.shape})"
        raise ValueError(msg)

    path = file_io.Path(path)

    # Check whether the format can be inferred from the filename
    if format is None:
        match path.suffix.lower():
            case ".pth", ".pt":
                format = LabelsFormat.TORCH
            case ".safetensors":
                format = LabelsFormat.SAFETENSORS
            case _:
                msg = f"Could not infer labels format from path: {path}"
                raise ValueError(msg)

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the labels
    match LabelsFormat(format):
        case LabelsFormat.SAFETENSORS:
            assert (
                path.suffix.lower() == ".safetensors"
            ), f"Expected SAFETENSORS file, got {path}"
            safetensors.save_file({"data": torch.as_tensor(self)}, path)
        case LabelsFormat.TORCH:
            assert path.suffix.lower() in (
                ".pth",
                ".pt",
            ), f"Expected PT file, got {path}"
            torch.save(torch.as_tensor(self), path)
        case LabelsFormat.PNG_UINT16 | LabelsFormat.PNG_L16:
            assert path.suffix.lower() == ".png", f"Expected PNG file, got {path}"
            sem, ins = PanopticMap.to_parts(self, as_tuple=True)

            void_id = kwargs.get("void_id", 64)
            max_ins = kwargs.get("divisor", 1000)

            assert not torch.any(
                sem == void_id
            ), f"Found void ID in semantic map, {void_id=}"
            assert torch.all(ins < max_ins), f"Found instance ID >= {max_ins=}"

            pan = sem.where(sem >= 0, void_id) * max_ins + ins
            write_png_l16(path, pan)
        case LabelsFormat.CITYSCAPES:
            divisor = kwargs.get("divisor", 1000)
            void_id = kwargs.get("void_id", 255)
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self % _BYTE_OFFSET
            img[:, :, 1] = self // _BYTE_OFFSET
            img[:, :, 2] = self // _BYTE_OFFSET // _BYTE_OFFSET
            img = torch.where(self == void_id * divisor, 0, img)
            img = torch.where(self < divisor, img * divisor, img - 1)

            write_png_rgb(path, img)
        case LabelsFormat.CITYSCAPES_VPS:
            divisor = 1000
            void_id = 255
            sem, ids = PanopticMap.to_parts(self, as_tuple=True)
            img = torch.where(ids > 0, ids - 1 + sem * divisor, sem)
            img = torch.where(img == void_id, 0, img)

            write_png_l16(path, img)
        case LabelsFormat.CITYSCAPES_DVPS:
            # https://github.com/joe-siyuan-qiao/ViP-DeepLab/tree/master/cityscapes-dvps
            max_ins = 255
            void_id = 32

            sem, ids = PanopticMap.to_parts(self.detach().cpu(), as_tuple=True)
            assert (sem < void_id).all(), sem.unique().tolist()
            sem[sem < 0] = void_id
            assert (ids < max_ins).all(), ids.unique().tolist()

            img = torch.zeros((*self.shape[-2:], 3), dtype=torch.uint8)
            img[:, :, 0] = sem
            img[:, :, 1] = ids

            write_png_rgb(path, img)
        case LabelsFormat.KITTI:  # KITTI STEP
            img = torch.empty((*self.shape[-2:], 3), dtype=torch.uint8)

            sem, ids = PanopticMap.to_parts(self, as_tuple=True)
            img[:, :, 0] = sem
            img[:, :, 1] = ids // _BYTE_OFFSET
            img[:, :, 2] = ids % _BYTE_OFFSET

            write_png_rgb(path, img)
        case LabelsFormat.VISTAS:
            divisor = 1000
            void_id = 255
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self % _BYTE_OFFSET
            img[:, :, 1] = self // _BYTE_OFFSET
            img[:, :, 2] = self // _BYTE_OFFSET // _BYTE_OFFSET

            write_png_rgb(path, img)
        case LabelsFormat.WILD_DASH:
            divisor = int(1e8)
            void_id = 255
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self // (_BYTE_OFFSET * _BYTE_OFFSET)
            img[:, :, 1] = (self // _BYTE_OFFSET) % _BYTE_OFFSET
            img[:, :, 2] = self % _BYTE_OFFSET
            img = torch.where(self == void_id * divisor, 0, img)

            write_png_rgb(path, img)
        case _:
            msg = f"Could not save labels to {path!r} ({format=})"
            raise NotImplementedError(msg)
