from __future__ import annotations

import typing as T
from enum import StrEnum, auto

import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from torchvision.tv_tensors import Mask as _Mask
from typing_extensions import override

from unipercept import file_io
from unipercept.data.tensors.helpers import write_png
from unipercept.data.types.coco import COCOResultPanoptic, COCOResultPanopticSegment
from unipercept.utils.typings import Pathable

from .registry import pixel_maps

__all__ = ["PanopticMap", "LabelsFormat"]

if T.TYPE_CHECKING:
    from ..sets import Metadata

BYTE_OFFSET: T.Final[T.Literal[256]] = 256


class LabelsFormat(StrEnum):
    """
    Enumerates the different formats of labels that are supported. Uses the name of
    the dataset that introduced the format.
    """

    CITYSCAPES = auto()
    CITYSCAPES_VPS = auto()
    KITTI = auto()
    VISTAS = auto()
    WILD_DASH = auto()
    TORCH = auto()
    SAFETENSORS = auto()


@pixel_maps.register
class PanopticMap(_Mask):
    """
    Implements a panoptic segmentation map, where each pixel has the value:
        category_id * label_divisor + instance_id.
    """

    DIVISOR: T.ClassVar[int] = int(2**15)  # same for all datasets
    IGNORE: T.ClassVar[int] = -1

    @classmethod
    @torch.no_grad()
    def read(cls, path: Pathable, info: Metadata | None, /, **meta_kwds) -> T.Self:
        """Read a panoptic map from a file."""
        from .helpers import get_kwd, read_pixels

        panoptic_format = get_kwd(meta_kwds, "format", LabelsFormat)
        path = file_io.get_local_path(str(path))

        match panoptic_format:
            case LabelsFormat.SAFETENSORS:
                assert info is not None
                labels = safetensors.load_file(path)["data"].as_subclass(cls)
                assert labels is not None
                labels.translate_semantic_(translation=info["translations_dataset"])
            case LabelsFormat.TORCH:
                assert info is not None
                labels = torch.load(path, map_location=torch.device("cpu")).as_subclass(
                    cls
                )
                assert labels is not None
                assert isinstance(labels, (PanopticMap, torch.Tensor)), type(labels)
                labels = labels.as_subclass(cls)
                labels.translate_semantic_(translation=info["translations_dataset"])
            case LabelsFormat.CITYSCAPES:
                assert info is not None
                divisor = info["label_divisor"]
                ignore_label = info["ignore_label"]
                img = read_pixels(path, color=True)

                map_ = (
                    img[:, :, 0]
                    + BYTE_OFFSET * img[:, :, 1]
                    + BYTE_OFFSET * BYTE_OFFSET * img[:, :, 2]
                )
                map_ = torch.where(map_ > 0, map_, ignore_label)
                map_ = torch.where(map_ < divisor, map_ * divisor, map_ + 1)

                labels = cls.from_combined(map_, divisor)
                labels.translate_semantic_(
                    translation=info["translations_dataset"],
                )
            case LabelsFormat.CITYSCAPES_VPS:
                assert info is not None
                divisor = info["label_divisor"]
                ignore_label = info["ignore_label"]

                img = read_pixels(path, color=False)
                has_instance = img >= divisor

                ids = torch.where(has_instance, (img % divisor) + 1, 0)
                sem = torch.where(has_instance, img // divisor, img)
                sem[sem == ignore_label] = -1

                labels = cls.from_parts(sem, ids)

            case LabelsFormat.KITTI:
                assert info is not None
                # KITTI STEP used as reference

                img = read_pixels(path, color=True)
                sem = img[:, :, 0]  # R-channel
                ids = torch.add(
                    img[:, :, 1] * BYTE_OFFSET,  # G channel
                    img[:, :, 2],  # B channel
                )

                labels = cls.from_parts(sem, ids)
                labels.translate_semantic_(info.translations_dataset)
            case LabelsFormat.VISTAS:
                assert info is not None
                divisor = info["label_divisor"]

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

                labels = cls.from_combined(img, divisor)
                labels.translate_semantic_(translation=info["translations_dataset"])
            case LabelsFormat.WILD_DASH:
                annotations = get_kwd(meta_kwds, "annotations", list[dict[str, Any]])

                assert info is not None
                divisor = info["label_divisor"]
                ignore_label = info["ignore_label"]
                translations = info["stuff_translations"]

                img = read_pixels(path, color=True)
                img = (
                    img[:, :, 0].to(torch.long) * BYTE_OFFSET * BYTE_OFFSET
                    + img[:, :, 1].to(torch.long) * BYTE_OFFSET
                    + img[:, :, 2].to(torch.long)
                )
                sem = torch.full_like(img, ignore_label, dtype=torch.long)
                for ann in annotations:
                    id = ann["id"]
                    category_id = ann["category_id"]
                    mask = img == id
                    sem[mask] = category_id

                ids = torch.full_like(img, 0, dtype=torch.long)  # TODO

                labels = cls.from_parts(sem, ids)
                labels.translate_semantic_(
                    translation=translations,
                )
            case _:
                raise NotImplementedError(f"{panoptic_format=}")

        assert labels.ndim == 2, f"Expected 2D tensor, got {labels.ndim}D tensor"

        assert (
            labels is not None
        ), f"No labels were read from '{path}' (format: {panoptic_format})"

        if len(meta_kwds) > 0:
            raise TypeError(f"Unexpected keyword arguments: {tuple(meta_kwds.keys())}")

        labels.remove_instances_(info.background_ids)

        return labels

    def save(self, path: Pathable, format: LabelsFormat | str | None = None) -> None:
        """
        Save the panoptic map to a file.
        """
        from .helpers import get_kwd

        path = file_io.Path(path)
        if format is None:
            match path.suffix.lower():
                case ".pth", ".pt":
                    format = LabelsFormat.TORCH
                case ".safetensors":
                    format = LabelsFormat.SAFETENSORS
                case _:
                    msg = f"Could not infer labels format from path: {path}"
                    raise ValueError(msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        match LabelsFormat(format):
            case LabelsFormat.SAFETENSORS:
                safetensors.save_file({"data": torch.as_tensor(self)}, path)
            case LabelsFormat.TORCH:
                torch.save(torch.as_tensor(self), path)
            case LabelsFormat.CITYSCAPES:
                divisor = self.DIVISOR
                ignore_label = self.IGNORE
                img = torch.empty((*self.shape, 3), dtype=torch.uint8)
                img[:, :, 0] = self % BYTE_OFFSET
                img[:, :, 1] = self // BYTE_OFFSET
                img[:, :, 2] = self // BYTE_OFFSET // BYTE_OFFSET
                img = torch.where(self == ignore_label * divisor, 0, img)
                img = torch.where(self < divisor, img * divisor, img - 1)

                write_png(path, img)
            case LabelsFormat.CITYSCAPES_VPS:
                divisor = self.DIVISOR
                ignore_label = self.IGNORE
                sem, ids = self.to_parts(as_tuple=True)
                img = torch.where(ids > 0, ids - 1 + sem * divisor, sem)
                img = torch.where(img == ignore_label, 0, img)

                write_png(path, img)
            case LabelsFormat.KITTI:
                img = torch.empty((*self.shape, 3), dtype=torch.uint8)

                sem, ids = self.to_parts(as_tuple=True)
                img[:, :, 0] = sem
                img[:, :, 1] = ids // BYTE_OFFSET
                img[:, :, 2] = ids % BYTE_OFFSET

                write_png(path, img)
            case LabelsFormat.VISTAS:
                divisor = self.DIVISOR
                img = torch.empty((*self.shape, 3), dtype=torch.uint8)
                img[:, :, 0] = self % BYTE_OFFSET
                img[:, :, 1] = self // BYTE_OFFSET
                img[:, :, 2] = self // BYTE_OFFSET // BYTE_OFFSET

                write_png(path, img)
            case LabelsFormat.WILD_DASH:
                divisor = self.DIVISOR
                ignore_label = self.IGNORE
                img = torch.empty((*self.shape, 3), dtype=torch.uint8)
                img[:, :, 0] = self // (BYTE_OFFSET * BYTE_OFFSET)
                img[:, :, 1] = (self // BYTE_OFFSET) % BYTE_OFFSET
                img[:, :, 2] = self % BYTE_OFFSET
                img = torch.where(self == ignore_label * divisor, 0, img)

                write_png(path, img)
            case _:
                raise NotImplementedError(f"{format=}")

    @classmethod
    def default(cls, shape: torch.Size, device: torch.device | str = "cpu") -> T.Self:
        return torch.full(
            shape, cls.IGNORE * cls.DIVISOR, dtype=torch.long, device=device
        ).as_subclass(cls)

    @classmethod
    def default_like(cls, other: torch.Tensor) -> T.Self:
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
        tensor: torch.Tensor,
    ) -> T.Self:
        return tensor.to(dtype=torch.long, non_blocking=True).as_subclass(cls)

    @classmethod
    def from_parts(
        cls, semantic: torch.Tensor | T.Any, instance: torch.Tensor | T.Any
    ) -> T.Self:
        """
        Create an instance from a semantic segmentation and instance segmentation map by combining them
        using the global ``LABEL_DIVISOR``.
        """

        semantic = torch.as_tensor(semantic)
        instance = torch.as_tensor(instance)

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

        return panoptic.as_subclass(cls)

    @classmethod
    def from_combined(cls, encoded_map: torch.Tensor | T.Any, divisor: int) -> T.Self:
        """
        Decompose an encoded map into a semantic segmentation and instance segmentation map, then combine
        again using the global ``LABEL_DIVISOR``.
        """
        encoded_map = torch.as_tensor(encoded_map)
        assert encoded_map.dtype in (torch.int32, torch.int64), encoded_map.dtype

        return cls.from_parts(encoded_map // divisor, encoded_map % divisor)

    @T.overload
    def to_parts(self, as_tuple=False) -> torch.Tensor:
        ...

    @T.overload
    def to_parts(self, as_tuple=True) -> T.Tuple[torch.Tensor, torch.Tensor]:
        ...

    def to_parts(
        self, as_tuple: bool = False
    ) -> torch.Tensor | T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the semantic and instance segmentation maps, returing a tensor of size [..., 2].
        The first channel contains the semantic segmentation map, the second channel contains the instance
        id that is NOT UNIQUE for each class.
        """
        ignore_mask = self == PanopticMap.IGNORE
        sem = torch.floor_divide(self, self.DIVISOR).as_subclass(_Mask)
        ins = torch.remainder(self, self.DIVISOR).as_subclass(_Mask)
        ins[ignore_mask] = 0
        if as_tuple:
            return sem, ins
        return torch.stack((sem, ins), dim=-1)

    def get_semantic_map(self) -> _Mask:
        return torch.floor_divide(self, self.DIVISOR).as_subclass(_Mask)

    def get_semantic_masks(self) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each semantic class."""
        sem_map = self.get_semantic_map()
        uq = torch.unique(sem_map)
        yield from (
            (int(u), (sem_map == u).as_subclass(_Mask)) for u in uq if u != self.IGNORE
        )

    def get_semantic_mask(self, class_id: int) -> _Mask:
        """Return a mask for the specified semantic class."""
        return (self.get_semantic_map() == class_id).as_subclass(_Mask)

    def unique_semantics(self) -> torch.Tensor:
        """Count the number of unique semantic classes."""
        uq = torch.unique(self.get_semantic_map())
        return uq[uq >= 0]

    def get_instance_map(self) -> _Mask:
        # old: does not support same sub-id for different classes
        ins_ids = torch.remainder(self, self.DIVISOR)
        return torch.where(
            (ins_ids > 0) & (self != PanopticMap.IGNORE), self, 0
        ).as_subclass(_Mask)

    def get_instance_masks(self) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each instance."""
        ins_map = self.get_instance_map()
        uq = torch.unique(ins_map)
        yield from ((int(u), (ins_map == u).as_subclass(_Mask)) for u in uq if u > 0)

    def get_instance_mask(self, instance_id: int) -> _Mask:
        """Return a mask for the specified instance."""
        return (self.get_instance_map() == instance_id).as_subclass(_Mask)

    @T.overload
    def get_masks(
        self, with_void=False, return_label=True
    ) -> T.List[T.Tuple[int, _Mask]]:
        ...

    @T.overload
    def get_masks(
        self, with_void=False, return_label=False
    ) -> T.List[T.Tuple[int, _Mask]]:
        ...

    def get_masks(
        self, with_void: bool = False, return_label: bool = False
    ) -> T.List[T.Tuple[int, int, _Mask]] | T.List[T.Tuple[int, _Mask]]:
        """Return a mask for each semantic class and instance (if any)."""

        pan_map = self.unique()

        result = []

        for pan_id in pan_map:
            sem_id = pan_id // self.DIVISOR

            if sem_id == self.IGNORE:
                if not with_void:
                    continue
                ins_id = 0
            else:
                ins_id = torch.remainder(pan_id, self.DIVISOR)

            mask = (self == pan_id).as_subclass(_Mask)

            if not return_label:
                result.append((int(sem_id), int(ins_id), mask))
            else:
                result.append((int(pan_id), mask))
        return result

    def unique_instances(self) -> torch.Tensor:
        """Count the number of unique instances for each semantic class."""
        ins_mask = self.get_instance_map() != PanopticMap.IGNORE
        return torch.unique(self[ins_mask])

    def remove_instances_(self, semantic_list: T.Iterable[int]) -> None:
        """Remove instances for the specified semantic classes."""
        sem_map, ins_map = self.to_parts(as_tuple=True)

        # Compute candidate map where all pixels that are not in the semantic list are set to -1
        can_map = torch.where(ins_map > 0, sem_map, PanopticMap.IGNORE)

        # Set all pixels that are not in the semantic list to 0
        for class_ in semantic_list:
            self[can_map == class_] = class_ * self.DIVISOR

    def translate_semantic_(
        self, translation: dict[int, int], inverse: bool = False
    ) -> None:
        """
        Apply a translation to the class labels. The translation is a dictionary mapping old class IDs to
        new class IDs. All old class IDs that are not in the dictionary are mapped to ``ignore_label``.
        """

        sem_map, ins_map = self.to_parts(as_tuple=True)

        self.fill_(self.IGNORE)

        for (
            old_id,
            new_id,
        ) in translation.items():
            if inverse:
                old_id, new_id = new_id, old_id

            mask = sem_map == old_id
            self[mask] = new_id * self.DIVISOR + ins_map[mask]

    def get_nonempty(self) -> _Mask:
        """Return a new instance with only the non-empty pixels."""
        return self[self != self.IGNORE * self.DIVISOR].as_subclass(_Mask)

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
    def must_be_semantic_map(cls, t: torch.Tensor):
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
    def is_semantic_map(cls, t: torch.Tensor) -> bool:
        try:
            cls.must_be_semantic_map(t)
        except ValueError:
            return False
        return True

    @classmethod
    def must_be_instance_map(cls, t: torch.Tensor):
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
    def is_instance_map(cls, t: torch.Tensor) -> bool:
        try:
            cls.must_be_instance_map(t)
        except ValueError:
            return False
        return True


# def transform_label_map(label_map: torch.Tensor, transform: Transform) -> PanopticMap:
#     map_uint8 = np.zeros((*label_map.shape, 3), dtype=np.uint8)
#     map_uint8[:, :, 0] = label_map % BYTE_OFFSET
#     map_uint8[:, :, 1] = label_map // BYTE_OFFSET
#     map_uint8[:, :, 2] = label_map // BYTE_OFFSET // BYTE_OFFSET
#     map_tf = transform.apply_segmentation(map_uint8).astype(label_map.dtype)

#     return (map_tf[:, :, 0] + map_tf[:, :, 1] * BYTE_OFFSET + map_tf[:, :, 2] * BYTE_OFFSET * BYTE_OFFSET,)


# def label_map_to_image(label_map: torch.Tensor) -> NP.NDArray[np.uint8]:
#     map_uint8 = np.zeros((*label_map.shape, 3), dtype=np.uint8)
#     map_uint8[:, :, 0] = label_map % BYTE_OFFSET
#     map_uint8[:, :, 1] = label_map // BYTE_OFFSET
#     map_uint8[:, :, 2] = label_map // BYTE_OFFSET // BYTE_OFFSET

#     return map_uint8
