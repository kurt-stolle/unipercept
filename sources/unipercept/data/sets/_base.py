"""Implements a baseclass for the UniPercept dataset common format."""

from __future__ import annotations

import dataclasses
import dataclasses as D
import enum as E
import functools
import typing as T

import torch
import typing_extensions as TX

from unipercept.data.tensors import PanopticMap
from unipercept.utils.camera import build_calibration_matrix
from unipercept.utils.catalog import DataManager
from unipercept.utils.dataset import Dataset as _BaseDataset
from unipercept.utils.dataset import _Dataqueue
from unipercept.utils.frozendict import frozendict
from unipercept.utils.tensorclass import Tensorclass

if T.TYPE_CHECKING:
    from unipercept.data.collect import ExtractIndividualFrames
    from unipercept.model import CaptureData
    from unipercept.data.collect import QueueGeneratorType

from unipercept.data.types import COCOCategory, Manifest, QueueItem

__all__ = [
    "PerceptionDataset",
    "PerceptionDataqueue",
    "Metadata",
    "SClass",
    "SType",
    "StuffMode",
    "create_metadata",
    "catalog",
]

PerceptionDataqueue: T.TypeAlias = _Dataqueue[QueueItem]

# ---------------- #
# HELPER FUNCTIONS #
# ---------------- #


def _individual_frames_queue() -> ExtractIndividualFrames:
    from unipercept.data.collect import ExtractIndividualFrames

    return ExtractIndividualFrames()


# ---------------- #
# CANONICAL COLORS #
# ---------------- #


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
        else:
            if self.name == _CANONICAL_ROOT_NAME:
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


class StuffMode(E.Enum):
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


@D.dataclass(frozen=True, kw_only=True, unsafe_hash=True)
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

    # Dataset ID -> Sem. ID
    translations_dataset: frozendict[int, int]

    # Sem. ID -> Dataset ID
    translations_semantic: frozendict[int, int]

    @property
    def thing_o2e(self) -> T.Dict[int, int]:
        """
        Returns a mapping of dataset IDs to embedded IDs for object classes.
        """
        return self.thing_offsets

    @property
    def stuff_o2e(self) -> T.Dict[int, int]:
        """
        Returns a mapping of dataset IDs to embedded IDs for semantic classes.
        """
        return self.stuff_offsets

    @functools.cached_property
    def thing_e2o(self) -> T.Dict[int, int]:
        """
        Returns a mapping of embedded IDs to dataset IDs for object classes.
        """
        return {v: k for k, v in self.thing_offsets.items()}

    @functools.cached_property
    def stuff_e2o(self) -> dict[int, int]:
        """
        Returns a mapping of embedded IDs to dataset IDs for semantic classes.
        """
        return {v: k for k, v in self.stuff_offsets.items()}

    @property
    def depth_fixed(self) -> T.Dict[int, float]:
        """
        Returns a mapping of semantic IDs to fixed depths.
        """
        return {
            sem_id: sem_cls.depth_fixed
            for sem_id, sem_cls in self.semantic_classes.items()
            if sem_cls.depth_fixed is not None
        }

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
    def things(self) -> T.Tuple[SClass, ...]:
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
    def stuff(self) -> T.Tuple[SClass, ...]:
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
        return StuffMode.ALL_CLASSES == self.stuff_mode

    @property
    @TX.deprecated("Use `stuff_mode` instead.")
    def stuff_with_things(self) -> bool:
        """Deprecated."""
        return StuffMode.WITH_THING == self.stuff_mode

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
    def thing_classes(self) -> T.Tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.thing_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].name` instead.")
    def stuff_classes(self) -> T.Tuple[str, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].name for sem_id in self.stuff_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].color` instead.")
    def thing_colors(self) -> T.Tuple[RGB, ...]:
        """Deprecated."""
        return tuple(self.semantic_classes[sem_id].color for sem_id in self.thing_ids)

    @property
    @TX.deprecated("Use `semantic_classes[...].color` instead.")
    def stuff_colors(self) -> T.Tuple[RGB, ...]:
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


class SType(E.Flag):
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


def create_metadata(
    sem_seq: T.Sequence[SClass],
    *,
    depth_max: float,
    label_divisor: int = 1000,
    ignore_depth: float = 0.0,
    ignore_label: int = 255,
    fps: float = 17.0,
    stuff_mode: StuffMode = StuffMode.ALL_CLASSES,
) -> Metadata:
    """Generate dataset metadata object."""

    # Sort the list of classes such that stuff classes come first, then things
    sem_seq = sorted(
        sem_seq,
        key=lambda c: (int(1e6) if c.get("is_thing") else 0) + c["unified_id"],
    )

    # Automatically resolves any many-to-one mappings for semantic IDs (only one semantic ID per metadata will remain)
    sem_map = {c["unified_id"]: c for c in sem_seq}

    # Offsets are the embedded channel index for either stuff or things
    stuff_offsets: dict[int, int] = {}
    for sem_id, sem_cls in sem_map.items():
        if not sem_cls.is_stuff:
            continue
        else:
            stuff_offsets[sem_id] = len(stuff_offsets)

    # Add a special thing class to the segmentation, which is used to mask out thing classes while detecting stuff
    # classes
    if stuff_mode == StuffMode.WITH_THING:
        for sem_id in tuple(stuff_offsets.keys()):
            stuff_offsets[sem_id] += 1

    thing_offsets: dict[int, int] = {}
    for sem_id, sem_cls in sem_map.items():
        if not sem_cls.is_thing:
            continue
        thing_offsets[sem_id] = len(thing_offsets)

        if stuff_mode == StuffMode.ALL_CLASSES:
            stuff_offsets[sem_id] = len(stuff_offsets)
        elif stuff_mode == StuffMode.WITH_THING:
            stuff_offsets[sem_id] = 0

    return Metadata(
        fps=fps,
        depth_max=depth_max,
        ignore_label=ignore_label,
        label_divisor=label_divisor,
        stuff_mode=stuff_mode,
        translations_dataset=frozendict(
            {c["dataset_id"]: c["unified_id"] for c in sem_seq}
        ),
        translations_semantic=frozendict(
            {c["unified_id"]: c["dataset_id"] for c in sem_seq}
        ),
        stuff_offsets=frozendict(stuff_offsets),
        thing_offsets=frozendict(thing_offsets),
        semantic_classes=frozendict(sem_map),
    )


# ------------------ #
# DATASET MANAGEMENT #
# ------------------ #

catalog: DataManager["PerceptionDataset", Metadata] = DataManager()


class PerceptionDataset(
    _BaseDataset[Manifest, QueueItem, Tensorclass, Metadata],
):
    """Baseclass for datasets that are composed of captures and motions."""

    queue_fn: T.Callable[[Manifest], QueueGeneratorType] = dataclasses.field(
        default_factory=_individual_frames_queue
    )

    @TX.override
    def __init_subclass__(cls, id: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith("_"):
            if id is not None:
                id_canon = catalog.parse_key(id)
                if id != id_canon:
                    raise ValueError(
                        f"Directly specifying an ID that is not canonical not allowed: '{id}' should be '{id_canon}'!"
                    )
                catalog.register_dataset(id, info=cls._create_info)(
                    cls
                )  # is a decorator
        elif id is not None:
            msg = f"Classes starting with '_' should not have an ID, got: {id}"
            raise ValueError(msg)

        cls._data_cache = {}

    @classmethod
    def _load_capture_data(
        cls, sources: T.Sequence[CaptureSources], info: Metadata
    ) -> CaptureData:
        from unipercept.data.io import read_depth_map, read_image, read_segmentation
        from unipercept.data.tensors.helpers import multi_read
        from unipercept.model import CaptureData

        num_caps = len(sources)
        times = torch.linspace(0, num_caps / info["fps"], num_caps)

        cap_data = CaptureData(
            times=times,
            images=multi_read(read_image, "image", no_entries="error")(sources),
            segmentations=multi_read(read_segmentation, "panoptic", no_entries="none")(
                sources, info
            ),
            depths=multi_read(read_depth_map, "depth", no_entries="none")(sources),
            batch_size=[num_caps],
        )

        if (
            info.depth_fixed is not None
            and cap_data.depths is not None
            and cap_data.segmentations is not None
        ):
            for i in range(num_caps):
                sem_seg = (
                    cap_data.segmentations[i]
                    .as_subclass(PanopticMap)
                    .get_semantic_map()
                )
                for cat, fixed in info.depth_fixed.items():
                    cap_data.depths[i][sem_seg == cat] = fixed * info.depth_max

        return cap_data

    @classmethod
    def _load_motion_data(
        cls, sources: T.Sequence[up.data.types.MotionSources], info: Metadata
    ) -> up.model.MotionData:
        raise NotImplementedError(f"{cls.__name__} does not implement motion sources!")

    _data_cache: T.ClassVar[dict[str, up.model.InputData]] = {}

    @classmethod
    @TX.override
    def _load_data(
        cls, key: str, item: QueueItem, info: Metadata
    ) -> up.model.InputData:
        from unipercept.model import CameraModel, InputData

        # Check for cache hit, should be a memmaped tensor
        # if key in cls._data_cache:
        #     return cls._data_cache[key].clone().contiguous()
        # types.utils.check_typeddict(item, QueueItem)
        # Captures
        item_caps = item["captures"]
        item_caps_num = len(item_caps)
        assert item_caps_num > 0

        data_caps = cls._load_capture_data(item_caps, info)

        # Motions
        if "motions" in item:
            item_mots = item["motions"]
            item_mots_num = len(item_mots)
            assert item_mots_num > 0
            data_mots = cls._load_motion_data(item_mots, info)
        else:
            data_mots = None

        # Camera
        item_camera = item["camera"]
        data_camera = CameraModel(
            matrix=build_calibration_matrix(
                focal_lengths=[item_camera["focal_length"]],
                principal_points=[item_camera["principal_point"]],
                orthographic=False,
            ),
            image_size=torch.as_tensor(item_camera["image_size"]),
            pose=torch.eye(4),
            batch_size=[],
        )

        # IDs: (sequence, frame)
        ids = torch.tensor([hash(item["sequence"]), item["frame"]], dtype=torch.long)

        input_data = InputData(
            ids=ids,
            captures=data_caps,
            motions=data_mots,
            cameras=data_camera,
            content_boxes=torch.cat(
                [
                    torch.tensor([0, 0], dtype=torch.int32),
                    data_camera.image_size.to(dtype=torch.int32),
                ]
            ),
            batch_size=[],
        )  # .memmap_()

        # cls._data_cache[key] = input_data

        return input_data  # .clone().contiguous()
