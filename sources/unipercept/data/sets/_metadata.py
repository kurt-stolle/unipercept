"""
Common classes and functions for working with descriptive metadata in Unipercept.
"""

from __future__ import annotations

import dataclasses as D
import enum as E
import functools
import typing as T
from pathlib import Path
from typing import Any, TypeAlias

import typing_extensions as TX

from unipercept.utils.frozendict import frozendict

if T.TYPE_CHECKING:
    from unipercept.data.types.coco import COCOCategory

__all__ = [
    "RGB",
    "Metadata",
    "SClass",
    "SType",
    "StuffOffsetMode",
    "HW",
    "BatchType",
    "ImageSize",
]

BatchType: TypeAlias = list[frozendict[str, Any]]
HW: TypeAlias = tuple[int, int]
PathType: TypeAlias = Path | str
OptionalPath: TypeAlias = PathType | None


class ImageSize(T.NamedTuple):
    height: int
    width: int


class SampleInfo(T.TypedDict, total=False):
    num_instances: frozendict[int, int]  # Mapping: (Dataset ID) -> (Num. instances)
    num_pixels: frozendict[int, int]  # Mapping: (Dataset ID) -> (Num. pixels)


@D.dataclass(slots=True, frozen=True, weakref_slot=False, unsafe_hash=True, eq=True)
class RGB:
    r: int
    g: int
    b: int

    def __iter__(self) -> T.Iterator[int]:
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

_CANONICAL_ROOT_NAME = "<ROOT>"


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
            if self.name != _CANONICAL_ROOT_NAME:
                raise ValueError(
                    f"Root category must have name {_CANONICAL_ROOT_NAME}."
                )
        elif self.name == _CANONICAL_ROOT_NAME:
            raise ValueError(
                f"Non-root category must not have name {_CANONICAL_ROOT_NAME}."
            )

    @property
    def is_root(self) -> bool:
        return self.parent is None


# ---------------- #
# DATASET METADATA #
# ---------------- #

DatasetID = T.NewType("DatasetID", int)  # The ID as given in the dataset
OffsetID = T.NewType(
    "OffsetID", int
)  # The ID as an index in the list of all valid IDs, unique for all classes
EmbeddedID = T.NewType(
    "EmbeddedID", int
)  # The ID as it is represented internally, not unique for things and stuff


class StuffOffsetMode(E.StrEnum):
    """
    Defines segmentation modes.

    Attributes
    ----------
    STUFF_ONLY : int
        Only background is considered `stuff`.
    ALL_CLASSES : int
        All classes are included in the segmentation, including `thing` classes.
    WITH_THING : int
        Add a special `thing` class to the segmentation, which is used to mask out `thing` classes.
    """

    STUFF_ONLY = E.auto()
    ALL_CLASSES = E.auto()
    WITH_THING = E.auto()


class SourceFormatSpecifications(T.TypedDict, total=False):
    """
    Describes the format of the source data.
    """

    object_detection: T.Mapping[str, T.Any]
    object_tracking: T.Mapping[str, T.Any]
    panoptic_segmentation: T.Mapping[str, T.Any]
    semantic_segmentation: T.Mapping[str, T.Any]
    instance_segmentation: T.Mapping[str, T.Any]
    monocular_depth: T.Mapping[str, T.Any]


@D.dataclass(frozen=True, kw_only=True, unsafe_hash=True)
class Metadata:
    """
    Implements common dataset metadata for Unipercept.

    The lack of a clear naming convention for the fields is due to the fact that
    the fields are inherited from the Detectron2's evaluators, the COCO API and
    various developers who built on top of them.
    """

    fps: float = D.field(
        default=0.0,
        metadata={
            "help": (
                "The frame rate of the dataset. "
                "If not provided, it is assumed to be 0, e.g. for image datasets."
            )
        },
    )
    depth_range: tuple[float, float] = D.field(
        default=(1.0, 100.0),
        metadata={
            "help": "The range of metric depth values that can be observed in the dataset original depth maps."
        },
    )
    semantic_classes: T.Mapping[int, SClass] = D.field(
        default_factory=dict,
        metadata={
            "help": (
                "Dictionary of semantic classes, mapping dataset IDs to class specifications."
            )
        },
    )
    stuff_offset_mode: StuffOffsetMode = D.field(
        default=StuffOffsetMode.ALL_CLASSES,
        metadata={
            "help": (
                "Defines the segmentation mode. "
                "STUFF_ONLY: Only background is considered `stuff`. "
                "ALL_CLASSES: All classes are included in the segmentation, including `thing` classes. "
                "WITH_THING: Add a special `thing` class to the segmentation, which is used to mask out `thing` classes."
            )
        },
    )
    source_specs: SourceFormatSpecifications = D.field(
        default_factory=dict,  # type: ignore
        metadata={
            "help": (
                "Dictionary of source format specification metadata for different data types as provided by "
                "the dataset. Sample-specific information should be stored in format "
                "field of that sample itself."
            )
        },
    )

    @classmethod
    def from_parameters(
        cls,
        sem_seq: T.Sequence[SClass],
        *,
        depth_max: float = 100.0,
        depth_min: float = 1.0,
        fps: float = 0.0,
        stuff_offset_mode: StuffOffsetMode = StuffOffsetMode.ALL_CLASSES,
        **kwargs,
    ) -> T.Self:
        """Generate dataset metadata object."""

        GENERATED_KEYS = {"fps", "depth_range", "stuff_offset_mode", "semantic_classes"}
        PROVIDED_KWARGS = set(kwargs.keys())
        if not PROVIDED_KWARGS.isdisjoint(GENERATED_KEYS):
            msg = f"Cannot provide keys: {PROVIDED_KWARGS - GENERATED_KEYS}"
            raise ValueError(msg)

        # Sort the list of classes such that stuff classes come first, then things
        sem_seq = sorted(
            sem_seq,
            key=lambda c: (int(1e6) if c.get("is_thing") else 0) + c["unified_id"],
        )

        # Automatically resolves any many-to-one mappings for semantic IDs (only one semantic ID per metadata will remain)
        sem_map = {c["unified_id"]: c for c in sem_seq}

        return cls(
            fps=fps,
            depth_range=(depth_min, depth_max),
            stuff_offset_mode=stuff_offset_mode,
            semantic_classes=frozendict(sem_map),
        )

    @property
    def depth_min(self) -> float:
        return self.depth_range[0]

    @property
    def depth_max(self) -> float:
        return self.depth_range[1]

    @property
    def translations_dataset(self) -> T.Mapping[int, int]:
        return {
            c["dataset_id"]: c["unified_id"] for c in self.semantic_classes.values()
        }

    @property
    def translations_semantic(self) -> T.Mapping[int, int]:
        return {
            c["unified_id"]: c["dataset_id"] for c in self.semantic_classes.values()
        }

    @property
    def stuff_offsets(self) -> T.Mapping[int, int]:
        stuff_offsets: dict[int, int] = {}
        for sem_id, sem_cls in self.semantic_classes.items():
            if not sem_cls.is_stuff:
                continue
            stuff_offsets[sem_id] = len(stuff_offsets)

        # Add a special thing class to the segmentation, which is used to mask out thing classes while detecting stuff
        # classes
        match self.stuff_offset_mode:
            case StuffOffsetMode.STUFF_ONLY:
                pass
            case StuffOffsetMode.ALL_CLASSES:
                stuff_offsets.update(
                    {
                        id: offset + len(stuff_offsets)
                        for id, offset in self.thing_offsets.items()
                    }
                )
            case StuffOffsetMode.WITH_THING:
                for sem_id in tuple(stuff_offsets.keys()):
                    stuff_offsets[sem_id] += 1
            case _:
                msg = f"Invalid stuff offset mode: {self.stuff_offset_mode}"
                raise ValueError(msg)
        return stuff_offsets

    @property
    def thing_offsets(self) -> T.Mapping[int, int]:
        thing_offsets: dict[int, int] = {}
        for sem_id, sem_cls in self.semantic_classes.items():
            if not sem_cls.is_thing:
                continue
            thing_offsets[sem_id] = len(thing_offsets)
        return thing_offsets

    @property
    def thing_o2e(self) -> T.Mapping[int, int]:
        """
        Returns a mapping of dataset IDs to embedded IDs for object classes.
        """
        return self.thing_offsets

    @property
    def stuff_o2e(self) -> T.Mapping[int, int]:
        """
        Returns a mapping of dataset IDs to embedded IDs for semantic classes.
        """
        return self.stuff_offsets

    @functools.cached_property
    def thing_e2o(self) -> T.Mapping[int, int]:
        """
        Returns a mapping of embedded IDs to dataset IDs for object classes.
        """
        return {v: k for k, v in self.thing_offsets.items()}

    @functools.cached_property
    def stuff_e2o(self) -> T.Mapping[int, int]:
        """
        Returns a mapping of embedded IDs to dataset IDs for semantic classes.
        """
        return {v: k for k, v in self.stuff_offsets.items()}

    @property
    def depth_fixed(self) -> T.Mapping[int, float]:
        """
        Returns a mapping of semantic IDs to fixed depths.
        """
        return {
            sem_id: sem_cls.depth_fixed
            for sem_id, sem_cls in self.semantic_classes.items()
            if sem_cls.depth_fixed is not None
        }

    def is_compatible(self, other: Metadata) -> bool:
        """
        Check whether this metadata is comatible with another metadata.
        This entails that the datasets have the same thing and stuff offsets.
        Note that the actual semantic classes do not need to be the same.
        This function only checks whether the amount of classes and their offsets are
        the same.
        """
        if len(set(self.translations_dataset.values())) != len(
            set(other.translations_dataset.values())
        ):
            return False
        if len(self.thing_offsets) != len(other.thing_offsets):
            return False
        if len(self.stuff_offsets) != len(other.stuff_offsets):
            return False
        return True

    # -------------- #
    # General access #
    # -------------- #

    @property
    def semantic_amount(self) -> int:
        """
        Returns the total amount of classes, including both thing and stuff classes.
        Duplicates are not counted.
        """
        classes = set(self.thing_offsets.keys()) | set(self.stuff_offsets.keys())
        return len(classes)

    # -------------- #
    # Thing specific #
    # -------------- #

    @property
    def thing_ids(self) -> frozenset[int]:
        """
        Returns the IDs of all object classes, i.e. those that can be is_thingly detected.
        """
        return frozenset(self.thing_offsets.keys())

    object_ids = thing_ids

    @property
    def thing_amount(self) -> int:
        """
        Returns the amount of **detectable** thing-type object classes. Defined as the amount of object classes that
        are not background (pure stuff) classes.
        """
        return len(self.thing_e2o)

    @functools.cached_property
    def things(self) -> tuple[SClass, ...]:
        """
        Returns the class specification of all is_thingly detectable object classes.
        """
        return tuple(self.semantic_classes[sem_id] for sem_id in self.thing_ids)

    # -------------- #
    # Stuff specific #
    # -------------- #

    @functools.cached_property
    def stuff_ids(self) -> frozenset[int]:
        """
        Returns the IDs of all semantic classes, which may include thing-type objects depending on the mode.
        """
        return frozenset(self.stuff_offsets.keys())

    @functools.cached_property
    def background_ids(self) -> frozenset[int]:
        """
        Returns the IDs of all background (pure stuff) classes.
        """
        return frozenset(self.stuff_ids - self.thing_ids)

    @functools.cached_property
    def stuff_amount(self) -> int:
        """
        Returns the amount of semantic classes, which may include thing-type objects depending on the segmentation mode.
        """
        return len(self.stuff_e2o)

    @functools.cached_property
    def stuff(self) -> tuple[SClass, ...]:
        """
        Returns the class specification of all semantic classes, which may include is_thing classes depending on the
        segmentation mode.
        """
        return tuple(self.semantic_classes[sem_id] for sem_id in self.stuff_ids)

    # ---------------- #
    # General metadata #
    # ---------------- #

    def __getitem__(self, key: str) -> T.Any:
        return getattr(self, key)

    def get(self, key: str, default: T.Any = None) -> T.Any:
        return getattr(self, key, default)

    # Implementation of deprecated properties from new metadata
    @property
    @TX.deprecated("Use `stuff_mode` instead.")
    def stuff_all_classes(self) -> bool:
        """Deprecated."""
        return self.stuff_offset_mode == StuffOffsetMode.ALL_CLASSES

    @property
    @TX.deprecated("Use `stuff_mode` instead.")
    def stuff_with_things(self) -> bool:
        """Deprecated."""
        return self.stuff_offset_mode == StuffOffsetMode.WITH_THING

    @property
    @TX.deprecated("Use `thing_amount` instead.")
    def num_thing(self) -> int:
        """Deprecated."""
        return len(self.thing_ids)

    @property
    @TX.deprecated("Use `stuff_amount` instead.")
    def num_stuff(self) -> int:
        """Deprecated."""
        return self.stuff_amount

    @property
    @TX.deprecated("Use `semantic_classes[...].name` instead.")
    def thing_classes(self) -> tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.thing_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].name` instead.")
    def stuff_classes(self) -> tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.stuff_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].color` instead.")
    def thing_colors(self) -> tuple[RGB, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].color for sem_id in self.thing_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].color` instead.")
    def stuff_colors(self) -> tuple[RGB, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].color for sem_id in self.stuff_ids)

    @property
    @TX.deprecated("Use `translations_dataset` instead.")
    def thing_translations(self) -> dict[int, int]:
        """Deprecated."""
        return {
            k: v for k, v in self.translations_dataset.items() if k in self.thing_ids
        }

    @property
    @TX.deprecated("Use `thing_offsets` instead.")
    def thing_embeddings(self) -> dict[int, int]:
        """Deprecated."""
        return self.thing_offsets

    @property
    @TX.deprecated("Use `translations_dataset` instead.")
    def stuff_translations(self) -> dict[int, int]:
        """Deprecated."""
        return {
            k: v for k, v in self.translations_dataset.items() if k in self.stuff_ids
        }

    @property
    @TX.deprecated("Use `stuff_offsets` instead.")
    def stuff_embeddings(self) -> dict[int, int]:
        """Deprecated."""
        return self.stuff_offsets


# --------------------------- #
# DATASET SEMANTIC CATEGORIES #
# --------------------------- #


class SType(E.IntFlag):
    VOID = 0
    STUFF = E.auto()
    THING = E.auto()


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
    depth_fixed: float | None = None

    @property
    def is_thing(self) -> bool:
        return SType.THING in self.kind

    @property
    def is_void(self) -> bool:
        return self.kind == SType.VOID

    @property
    def is_stuff(self) -> bool:
        return SType.STUFF in self.kind

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
