"""Implements a baseclass for the UniPercept dataset common format."""

from __future__ import annotations

import dataclasses
import dataclasses as D
import enum as E
import typing as T

import torch
from typing_extensions import deprecated, override
from unicore import catalog
from unicore.utils.dataset import Dataset as _BaseDataset
from unicore.utils.frozendict import frozendict

from unipercept.data.tensors import PanopticMap
from unipercept.utils.camera import build_calibration_matrix

if T.TYPE_CHECKING:
    import unipercept as up
    from unipercept.data.collect import ExtractIndividualFrames
    from unipercept.model import InputData

from ..types import COCOCategory, Manifest, QueueItem

__all__ = ["PerceptionDataset", "Metadata", "SClass", "SType", "StuffMode", "get_default_queue_fn", "info_factory"]

# ---------------- #
# HELPER FUNCTIONS #
# ---------------- #


def get_default_queue_fn() -> ExtractIndividualFrames:
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


class StuffMode(E.StrEnum):
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

    STUFF_ONLY = E.auto()
    ALL_CLASSES = E.auto()
    WITH_THING = E.auto()


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
    def depth_fixed(self) -> T.Dict[int, float]:
        """
        Returns a mapping of semantic IDs to fixed depths.
        """
        return {
            sem_id: sem_cls.depth_fixed
            for sem_id, sem_cls in self.semantic_classes.items()
            if sem_cls.depth_fixed is not None
        }

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


def info_factory(
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

    sem_seq = sorted(
        # filter(
        #     lambda c: c["unified_id"] >= 0 and not c.is_void,
        #     sem_seq,
        # ),
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

    if stuff_mode == StuffMode.WITH_THING:
        # Cast to tuple in order to not alter the original dict while iterating
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
        translations_dataset=frozendict({c["dataset_id"]: c["unified_id"] for c in sem_seq}),
        translations_semantic=frozendict({c["unified_id"]: c["dataset_id"] for c in sem_seq}),
        stuff_offsets=frozendict(stuff_offsets),
        thing_offsets=frozendict(thing_offsets),
        semantic_classes=frozendict(sem_map),
    )

    # num_thing = 0
    # num_stuff = 0
    # depth_fixed = {}
    # thing_classes = []
    # stuff_classes = []
    # thing_colors = []
    # stuff_colors = []
    # thing_translations = {}
    # thing_embeddings = {}
    # stuff_translations = {}
    # stuff_embeddings = {}
    # thing_train_id2contiguous_id = {}
    # stuff_train_id2contiguous_id = {}
    # cats_tracked = []

    # # Create definition of training IDs, which differ for stuff and things
    # for k in categories:
    #     dataset_id = k["id"]
    #     train_id = k["trainId"]
    #     if k["trainId"] == ignore_label:
    #         continue

    #     if bool(k.get("isthing")) == 1:
    #         id_duplicate = train_id in thing_embeddings
    #         thing_translations[dataset_id] = train_id

    #         if not id_duplicate:
    #             assert train_id not in stuff_embeddings, f"Train ID {train_id} is duplicated in stuff."

    #             thing_classes.append(k["name"])
    #             thing_colors.append(RGB(*k["color"]))
    #             thing_embeddings[train_id] = num_thing
    #             num_thing += 1

    #         if stuff_all_classes:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 stuff_embeddings[train_id] = num_stuff
    #                 num_stuff += 1
    #     else:
    #         id_duplicate = train_id in stuff_embeddings
    #         if not id_duplicate:
    #             stuff_classes.append(k["name"])
    #             stuff_colors.append(RGB(*k["color"]))
    #         if stuff_with_things and not stuff_all_classes:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 num_stuff += 1
    #                 stuff_embeddings[train_id] = num_stuff + 1
    #         else:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 stuff_embeddings[train_id] = num_stuff
    #                 num_stuff += 1

    # # Sanity check
    # assert 0 in thing_embeddings.values()
    # assert 0 in stuff_embeddings.values()

    # # Create train ID to color mapping
    # colors: dict[int, RGB] = {}
    # colors |= {k: thing_colors[v] for k, v in thing_embeddings.items()}
    # colors |= {k: stuff_colors[v] for k, v in stuff_embeddings.items()}

    # # Create inverse translations
    # for key, value in thing_embeddings.items():
    #     thing_train_id2contiguous_id[value] = key
    # for key, value in stuff_embeddings.items():
    #     stuff_train_id2contiguous_id[value] = key

    # return Metadata(
    #     colors=frozendict(colors),
    #     stuff_all_classes=stuff_all_classes,
    #     stuff_with_things=stuff_with_things,
    #     ignore_label=ignore_label,
    #     fps=fps,
    #     num_thing=num_thing,
    #     num_stuff=num_stuff,
    #     label_divisor=label_divisor,
    #     depth_max=depth_max,
    #     depth_fixed=frozendict(depth_fixed),
    #     thing_classes=tuple(thing_classes),
    #     stuff_classes=tuple(stuff_classes),
    #     thing_colors=tuple(thing_colors),
    #     stuff_colors=tuple(stuff_colors),
    #     thing_translations=frozendict(thing_translations),
    #     thing_embeddings=frozendict(thing_embeddings),
    #     stuff_translations=frozendict(stuff_translations),
    #     stuff_embeddings=frozendict(stuff_embeddings),
    #     thing_train_id2contiguous_id=frozendict(thing_train_id2contiguous_id),
    #     stuff_train_id2contiguous_id=frozendict(stuff_train_id2contiguous_id),
    #     cats_tracked=frozenset(thing_train_id2contiguous_id.values()),
    # )


# ----------------- #
# DATASET BASECLASS #
# ----------------- #


class PerceptionDataset(
    _BaseDataset[Manifest, QueueItem, "InputData", Metadata],
):
    """Baseclass for datasets that are composed of captures and motions."""

    queue_fn: T.Callable[[Manifest], up.data.collect.QueueGeneratorType] = dataclasses.field(
        default_factory=get_default_queue_fn
    )

    @override
    def __init_subclass__(cls, id: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith("_"):
            if id is not None:
                id_canon = catalog.canonicalize_id(id)
                if id != id_canon:
                    raise ValueError(
                        f"Directly specifying an ID that is not canonical not allowed: '{id}' should be '{id_canon}'!"
                    )

            catalog.register_dataset(id)(cls)  # is a decorator
        elif id is not None:
            raise ValueError(f"Opinionated: classes starting with '_' should not have an ID, got: {id}")

        cls._data_cache = {}

    @classmethod
    def _load_capture_data(
        cls, sources: T.Sequence[up.data.types.CaptureSources], info: Metadata
    ) -> up.model.CaptureData:
        from unipercept.data.io import read_depth_map, read_image, read_segmentation
        from unipercept.data.tensors.helpers import multi_read
        from unipercept.model import CaptureData

        num_caps = len(sources)
        times = torch.linspace(0, num_caps / info["fps"], num_caps)

        cap_data = CaptureData(
            times=times,
            images=multi_read(read_image, "image", no_entries="error")(sources),
            segmentations=multi_read(read_segmentation, "panoptic", no_entries="none")(sources, info),
            depths=multi_read(read_depth_map, "depth", no_entries="none")(sources),
            batch_size=[num_caps],
        )

        if info.depth_fixed is not None and cap_data.depths is not None and cap_data.segmentations is not None:
            for i in range(num_caps):
                sem_seg = cap_data.segmentations[i].as_subclass(PanopticMap).get_semantic_map()
                for cat, fixed in info.depth_fixed.items():
                    cap_data.depths[i][sem_seg == cat] = fixed * info.depth_max

        return cap_data

    @classmethod
    def _load_motion_data(
        cls, sources: T.Sequence[up.data.types.MotionSources], info: up.data.types.Metadata
    ) -> up.model.MotionData:
        raise NotImplementedError(f"{cls.__name__} does not implement motion sources!")

    _data_cache: T.ClassVar[dict[str, up.model.InputData]] = {}

    @classmethod
    @override
    def _load_data(cls, key: str, item: QueueItem, info: Metadata) -> up.model.InputData:
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
                [torch.tensor([0, 0], dtype=torch.int32), data_camera.image_size.to(dtype=torch.int32)]
            ),
            batch_size=[],
        )  # .memmap_()

        # cls._data_cache[key] = input_data

        return input_data  # .clone().contiguous()
