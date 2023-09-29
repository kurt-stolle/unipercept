"""
Types that describe semantic properties of visual objects.
"""

from __future__ import annotations

import dataclasses as D
import enum
import typing as T
from collections.abc import Iterator

from typing_extensions import deprecated, override
from unicore.utils.frozendict import frozendict

if T.TYPE_CHECKING:
    from ._coco import COCOCategory


@D.dataclass(slots=True, frozen=True, weakref_slot=False, unsafe_hash=True, eq=True)
class RGB:
    r: int
    g: int
    b: int

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the RGB values.
        """
        return iter(D.astuple(self))

    def __getitem__(self, idx: int) -> int:
        """
        Returns the RGB value at the given index.
        """
        return D.astuple(self)[idx]

    def __len__(self) -> int:
        """
        Returns the number of RGB values.
        """
        return 3


# -------------------- #
# CANONICAL CATEGORIES #
# -------------------- #

CANONICAL_ROOT_NAME = "<ROOT>"


@D.dataclass(slots=True, frozen=True, kw_only=True, weakref_slot=True, unsafe_hash=True)
class CanonicalClass:
    """
    Defines a canonical category ("class"), which is a artificial grouping of categories that are semantically similar.
    This also enables us to map categories from different datasets to the same category in the model.

    Parameters
    ----------
    name : str
        Name of the category.
    parent : CanonicalClass
        Parent category, if any. If None, this is the root category and the name must be ``CANONICAL_ROOT_NAME``.
    """

    name: str
    parent: CanonicalClass | None

    def __post_init__(self):
        if self.parent is None:
            if self.name != CANONICAL_ROOT_NAME:
                raise ValueError(f"Root category must have name {CANONICAL_ROOT_NAME}.")
        else:
            if self.name == CANONICAL_ROOT_NAME:
                raise ValueError(f"Non-root category must not have name {CANONICAL_ROOT_NAME}.")

    @property
    def is_root(self) -> bool:
        return self.parent is None


# ---------------- #
# DATASET METADATA #
# ---------------- #


class StuffMode(enum.StrEnum):
    """
    Defines segmentation modes.

    Attributes
    ----------
    BACKGROUND : int
        Only background is considered `stuff`.
    ALL_CLASSES : int
        All classes are included in the segmentation, including `thing` classes.
    WITH_THING : int
        Add a special `thing` class to the segmentation, which is used to mask out `thing` classes.
    """

    STUFF_ONLY = enum.auto()
    ALL_CLASSES = enum.auto()
    WITH_THING = enum.auto()


# class Metadata(T.TypedDict):
@D.dataclass(slots=True, frozen=True, kw_only=True, weakref_slot=True, unsafe_hash=True)
class Metadata:
    """
    Implements common dataset metadata for Unipercept.

    The lack of a clear naming convention for the fields is due to the fact that
    the fields are inherited from the Detectron2's evaluators, the COCO API and
    various developers who built on top of them.
    """

    label_divisor: int
    ignore_label: int

    fps: float
    depth_max: float

    # Sem. ID -> Sem. Class
    semantic_classes: frozendict[int, SClass]

    # Sem. ID --> Embedded ID
    thing_offsets: frozendict[int, int]
    stuff_offsets: frozendict[int, int]
    stuff_mode: StuffMode

    # Dataset ID -> Sem. ID and Sem. ID -> Dataset ID
    translations_dataset: frozendict[int, int]
    translations_semantic: frozendict[int, int]

    @property
    def thing_ids(self) -> T.Tuple[int, ...]:
        """
        Returns the IDs of all object classes, i.e. those that can be is_thingly detected.
        """
        return tuple(self.thing_offsets.keys())

    @property
    def thing_amount(self) -> int:
        """
        Returns the amount of is_thingly detectable object classes.
        """
        return len(self.thing_ids)

    @property
    def things(self) -> T.Tuple[SClass, ...]:
        """
        Returns the class specification of all is_thingly detectable object classes.
        """
        return tuple(self.semantic_classes[sem_id] for sem_id in self.thing_ids)

    @property
    def stuff_ids(self) -> T.Tuple[int, ...]:
        """
        Returns the IDs of all semantic classes, which may include is_thing classes depending on the segmentation mode.
        """
        return tuple(self.stuff_offsets.keys())

    @property
    def stuff_amount(self) -> int:
        """
        Returns the amount of semantic classes, which may include is_thing classes depending on the segmentation mode.
        """
        return len(self.stuff_ids)

    @property
    def stuff(self) -> T.Tuple[SClass, ...]:
        """
        Returns the class specification of all semantic classes, which may include is_thing classes depending on the
        segmentation mode.
        """
        return tuple(self.semantic_classes[sem_id] for sem_id in self.stuff_ids)

    def __getitem__(self, key: str) -> T.Any:
        return getattr(self, key)

    def get(self, key: str, default: T.Any = None) -> T.Any:
        return getattr(self, key, default)

    # Legacy metadata (consider this deprecated)
    # stuff_all_classes: bool
    # stuff_with_things: bool
    # num_thing: int
    # num_stuff: int
    # thing_classes: tuple[str]
    # stuff_classes: tuple[str]
    # thing_colors: tuple[RGB]
    # stuff_colors: tuple[RGB]
    # thing_translations: frozendict[int, int]
    # thing_embeddings: frozendict[int, int]
    # stuff_translations: frozendict[int, int]
    # stuff_embeddings: frozendict[int, int]
    # thing_train_id2contiguous_id: frozendict[int, int]
    # stuff_train_id2contiguous_id: frozendict[int, int]

    # Implementation of deprecated properties from new metadata
    @property
    @deprecated("Use `stuff_mode` instead.")
    def stuff_all_classes(self) -> bool:
        """Deprecated."""
        return StuffMode.ALL_CLASSES in self.stuff_mode

    @property
    @deprecated("Use `stuff_mode` instead.")
    def stuff_with_things(self) -> bool:
        """Deprecated."""
        return StuffMode.WITH_THING in self.stuff_mode

    @property
    @deprecated("Use `thing_amount` instead.")
    def num_thing(self) -> int:
        """Deprecated."""
        return len(self.thing_ids)

    @property
    @deprecated("Use `stuff_amount` instead.")
    def num_stuff(self) -> int:
        """Deprecated."""
        return self.stuff_amount

    @property
    @deprecated("Use `semantic_classes[...].name` instead.")
    def thing_classes(self) -> T.Tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.thing_ids)

    @property
    @deprecated("Use `semantic_classes[...].name` instead.")
    def stuff_classes(self) -> T.Tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.stuff_ids)

    @property
    @deprecated("Use `semantic_classes[...].color` instead.")
    def thing_colors(self) -> T.Tuple[RGB, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].color for sem_id in self.thing_ids)

    @property
    @deprecated("Use `semantic_classes[...].color` instead.")
    def stuff_colors(self) -> T.Tuple[RGB, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].color for sem_id in self.stuff_ids)

    @property
    @deprecated("Use `translations_dataset` instead.")
    def thing_translations(self) -> dict[int, int]:
        """Deprecated."""
        return {k: v for k, v in self.translations_dataset.items() if k in self.thing_ids}

    @property
    @deprecated("Use `thing_offsets` instead.")
    def thing_embeddings(self) -> dict[int, int]:
        """Deprecated."""
        return self.thing_offsets

    @property
    @deprecated("Use `translations_dataset` instead.")
    def stuff_translations(self) -> dict[int, int]:
        """Deprecated."""
        return {k: v for k, v in self.translations_dataset.items() if k in self.stuff_ids}

    @property
    @deprecated("Use `stuff_offsets` instead.")
    def stuff_embeddings(self) -> dict[int, int]:
        """Deprecated."""
        return self.stuff_offsets

    @property
    @deprecated("Use `thing_offsets` instead.")
    def thing_train_id2contiguous_id(self) -> dict[int, int]:
        """Deprecated."""
        return {v: k for k, v in self.thing_offsets.items()}

    @property
    @deprecated("Use `stuff_offsets` instead.")
    def stuff_train_id2contiguous_id(self) -> dict[int, int]:
        """Deprecated."""
        return {v: k for k, v in self.stuff_offsets.items()}


# --------------------------- #
# DATASET SEMANTIC CATEGORIES #
# --------------------------- #


class SType(enum.Flag):
    VOID = 0
    STUFF = enum.auto()
    THING = enum.auto()


@D.dataclass(slots=True, frozen=True, kw_only=True, weakref_slot=True, unsafe_hash=True)
class SClass:
    """
    Defines a category ("class") in the dataset, following the format of the corresponding ground truth files provided
    by the distribution.
    """

    name: str
    color: RGB
    kind: SType
    unified_id: int
    dataset_id: int
    depth_fixed: T.Optional[float] = None

    @property
    def is_thing(self) -> bool:
        return self.kind == SType.THING

    @property
    def is_void(self) -> bool:
        return not self.is_thing and not self.is_stuff

    @property
    def is_stuff(self) -> bool:
        return self.kind == SType.STUFF

    def __getitem__(self, key: str) -> T.Any:
        return getattr(self, key)

    def get(self, key: str, default: T.Any = None) -> T.Any:
        return getattr(self, key, default)

    @classmethod
    def from_coco(cls, ccat: COCOCategory, **kwargs) -> SClass:
        """
        Converts a COCO category to a semantic class.
        """
        if ccat.get("isthing", 0) == 1:
            kind = SType.THING
        else:
            kind = SType.STUFF

        scls = cls(
            dataset_id=ccat["id"],
            unified_id=ccat["trainId"],
            color=RGB(*ccat["color"]),
            kind=kind,
            name=ccat["name"],
            **kwargs,
        )

        return scls

    def as_coco(self, ignore_label: int) -> COCOCategory:
        """
        Converts a semantic class to a COCO category.
        """
        unified_id = self.get("unified_id") or ignore_label
        ccat: COCOCategory = {
            "name": self["name"],
            "color": self["color"],
            "isthing": int(self["is_thing"]),
            "id": self["dataset_id"],
            "trainId": unified_id,
        }
        return ccat
