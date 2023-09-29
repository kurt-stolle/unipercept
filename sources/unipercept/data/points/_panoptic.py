from __future__ import annotations

import typing as T

import torch
from torchvision.datapoints import Mask as _Mask
from typing_extensions import override

from .registry import pixel_maps

__all__ = ["PanopticMap"]


@pixel_maps.register
class PanopticMap(_Mask):
    """
    Implements a panoptic segmentation map, where each pixel has the value:
        category_id * label_divisor + instance_id.
    """

    DIVISOR: T.ClassVar[int] = int(2**15)  # same for all datasets
    IGNORE: T.ClassVar[int] = -1

    @classmethod
    def default(cls, shape: torch.Size, device: torch.device | str = "cpu") -> T.Self:
        return torch.full(shape, cls.IGNORE * cls.DIVISOR, dtype=torch.int32, device=device).as_subclass(cls)

    @classmethod
    def default_like(cls, other: torch.Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(torch.full_like(other, fill_value=cls.IGNORE * cls.DIVISOR, dtype=torch.int32))

    @classmethod
    @override
    def wrap_like(
        cls,
        other: T.Self,
        tensor: torch.Tensor,
    ) -> T.Self:
        return tensor.to(dtype=torch.int32, non_blocking=True).as_subclass(cls)

    @classmethod
    def from_parts(cls, semantic: torch.Tensor, instance: torch.Tensor) -> T.Self:
        """
        Create an instance from a semantic segmentation and instance segmentation map by combining them
        using the global ``LABEL_DIVISOR``.
        """

        assert semantic.shape == instance.shape
        assert int(semantic.min()) >= 0 or int(semantic.min()) == cls.IGNORE
        assert int(instance.max()) <= cls.DIVISOR

        semantic = semantic.to(dtype=torch.int32, non_blocking=True)
        instance = instance.to(dtype=torch.int32, non_blocking=True)
        panoptic = instance + semantic * cls.DIVISOR

        return panoptic.as_subclass(cls)

    @classmethod
    def from_combined(cls, encoded_map: torch.Tensor, divisor: int) -> T.Self:
        """
        Decompose an encoded map into a semantic segmentation and instance segmentation map, then combine
        again using the global ``LABEL_DIVISOR``.
        """
        assert encoded_map.dtype in (torch.int32, torch.int64), encoded_map.dtype

        return cls.from_parts(encoded_map // divisor, encoded_map % divisor)

    def get_semantic_map(self) -> _Mask:
        return torch.floor_divide(self, self.DIVISOR).as_subclass(_Mask)

    def get_semantic_masks(self) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each semantic class."""
        sem_map = self.get_semantic_map()
        uq = torch.unique(sem_map)
        yield from ((int(u), (sem_map == u).as_subclass(_Mask)) for u in uq if u != self.IGNORE)

    def get_semantic_mask(self, class_id: int) -> _Mask:
        """Return a mask for the specified semantic class."""
        return (self.get_semantic_map() == class_id).as_subclass(_Mask)

    def unique_semantics(self) -> torch.Tensor:
        """Count the number of unique semantic classes."""
        uq = torch.unique(self.get_semantic_map())
        return uq[uq >= 0]

    def get_instance_map(self) -> _Mask:
        return torch.remainder(self, self.DIVISOR).as_subclass(_Mask)

    def get_instance_masks(self) -> T.Iterator[tuple[int, _Mask]]:
        """Return a list of masks, one for each instance."""
        ins_map = self.get_instance_map()
        uq = torch.unique(ins_map)
        yield from ((int(u), (ins_map == u).as_subclass(_Mask)) for u in uq)

    def get_instance_mask(self, instance_id: int) -> _Mask:
        """Return a mask for the specified instance."""
        return (self.get_instance_map() == instance_id).as_subclass(_Mask)

    def get_masks(self, with_void=False) -> T.Iterator[tuple[int, int, _Mask]]:
        """Return a mask for each semantic class and instance (if any)."""

        pan_map = self.unique()

        for pan_id in pan_map:
            sem_id = pan_id // self.DIVISOR
            ins_id = pan_id % self.DIVISOR

            if sem_id == self.IGNORE and not with_void:
                continue

            mask = (self == pan_id).as_subclass(_Mask)
            yield int(sem_id), int(ins_id), mask

    def unique_instances(self) -> torch.Tensor:
        """Count the number of unique instances for each semantic class."""
        ins_mask = self.get_instance_map() > 0
        return torch.unique(ins_mask) - 1

    def remove_instances_(self, semantic_list: list[int]) -> None:
        """Remove instances for the specified semantic classes."""
        sem_map = self.get_semantic_map()
        ins_map = self.get_instance_map()

        # Compute candidate map where all pixels that are not in the semantic list are set to -1
        can_map = torch.where(ins_map > 0, sem_map, -1)

        # Set all pixels that are not in the semantic list to 0
        for class_ in semantic_list:
            self[can_map == class_] = class_ * self.DIVISOR

    def translate_semantic_(self, translation: dict[int, int], ignore_label: int = IGNORE) -> None:
        """
        Apply a translation to the class labels. The translation is a dictionary mapping old class IDs to
        new class IDs. All old class IDs that are not in the dictionary are mapped to ``ignore_label``.
        """

        ins_map = self.get_instance_map()
        sem_map = self.get_semantic_map()

        self.fill_(ignore_label * self.DIVISOR)

        for (
            old_id,
            new_id,
        ) in translation.items():
            mask = sem_map == old_id
            self[mask] = new_id * self.DIVISOR + ins_map[mask]

    def get_nonempty(self) -> _Mask:
        """Return a new instance with only the non-empty pixels."""
        return self[self != self.IGNORE * self.DIVISOR].as_subclass(_Mask)
