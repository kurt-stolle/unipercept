r"""
Baseclass for all datasets in UniPercept.
"""

from __future__ import annotations

import copy
import dataclasses as D
import itertools
import typing as T
import warnings
from datetime import UTC, datetime
from pprint import pformat

import torch
import torch.utils.data

from unipercept.model import CaptureData, InputData, MotionData
from unipercept.utils.catalog import CatalogFromPackageMetadata
from unipercept.utils.dataset import Dataset as _BaseDataset
from unipercept.utils.dataset import _Datapipe, _Dataqueue
from unipercept.utils.tensorclass import Tensorclass

from ._manifest import (
    CaptureSources,
    Manifest,
    ManifestSequence,
    MotionSources,
    QueueItem,
)
from ._metadata import Metadata

__all__ = [
    "PerceptionDataset",
    "PerceptionDataqueue",
    "PerceptionDatapipe",
    "PerceptionGatherer",
    "ConcatenatedDataset",
    "SubsampledDataset",
    "catalog",
]

PerceptionDataqueue: T.TypeAlias = _Dataqueue["QueueItem"]
PerceptionDatapipe: T.TypeAlias = _Datapipe[
    "QueueItem",
    "InputData",
    "Metadata",
]
PerceptionGatherer: T.TypeAlias = T.Callable[[Manifest], tuple[str, QueueItem]]

catalog: CatalogFromPackageMetadata[PerceptionDataset, Metadata] = (
    CatalogFromPackageMetadata(group="unipercept.datasets")
)


class PerceptionDataset(
    _BaseDataset[Manifest, QueueItem, Tensorclass, Metadata],
):
    """Baseclass for datasets that are composed of captures and motions."""

    _VERSION_: T.ClassVar[str | None] = None
    _ID_: T.ClassVar[str | None] = None

    def download(self, *, force: bool = False) -> None:
        """
        Download the dataset to the local disk.
        """
        from unipercept.log import logger

        logger.warning("%s does not implement download!", self.__class__.__name__)

    @property
    def id(self) -> str:
        """
        Returns the ID of the dataset.
        """
        if self.__class__._ID_ is None:
            msg = f"{self.__class__.__name__} does not have an ID!"
            raise RuntimeError(msg)
        return self.__class__._ID_

    @property
    def version(self) -> str:
        """
        Returns the version of the dataset.
        """
        if self.__class__._VERSION_ is None:
            msg = f"{self.__class__.__name__} does not have a version!"
            raise RuntimeError(msg)
        return self.__class__._VERSION_

    @T.override
    def __init_subclass__(
        cls, id: str | None = None, version: str | None = "1.0", **kwargs
    ):
        super().__init_subclass__(**kwargs)

        if version is not None:
            if (version_canon := version.lower().strip()) != version:
                msg = (
                    f"Version {version!r} is not canonical! Expected: {version_canon!r}"
                )
                raise ValueError(msg)
            cls._VERSION_ = version

        if id is not None:
            if (id_canon := catalog.parse_key(id)) != id:
                msg = f"ID {id!r} is not canonical! Expected: {id_canon!r}"
                raise ValueError(msg)
            cls._ID_ = id

            # Register if an ID is provided
            catalog.register(id, info=cls._create_info)(cls)

        cls._data_cache = {}

    @classmethod
    def variants(cls) -> T.Iterator[dict[str, T.Any]]:
        """
        Returns a list of all possible variants of the dataset.
        """
        try:
            options = cls.options()
            for values in itertools.product(*options.values()):
                yield dict(zip(options.keys(), values, strict=False))
        except NotImplementedError:
            pass

    @classmethod
    def options(cls) -> dict[str, T.Iterable[T.Any]]:
        """
        Returns a dictionary of all possible keyword argument value options.
        """
        msg = f"{cls.__name__} does not implement get_variant_options!"
        raise NotImplementedError(msg)

    @classmethod
    def _load_capture_data(
        cls, sources: T.Sequence[CaptureSources], info: Metadata
    ) -> CaptureData:
        from unipercept.data import tensors
        from unipercept.data.tensors.helpers import multi_read
        from unipercept.model import CaptureData

        num_caps = len(sources)
        times = torch.linspace(0, num_caps / info["fps"], num_caps)

        # Check whether one of "panoptic" or "semantic"/"instance" is present in
        # the sources
        has_combined = any("panoptic" in src for src in sources)
        has_split = any("semantic" in src or "instance" in src for src in sources)
        if has_combined and has_split:
            msg = "Cannot have both panoptic and semantic/instance segmentation!"
            raise ValueError(msg)
        if has_combined:
            labels = multi_read(tensors.read_panoptic, "panoptic", no_entries="error")(
                sources
            )
        elif has_split:
            sem_seg = multi_read(tensors.read_mask, "semantic", no_entries="error")(
                sources
            )
            ins_seg = multi_read(tensors.read_mask, "instance", no_entries="none")(
                sources
            )
            if ins_seg is None:
                labels = tensors.PanopticMap.from_semantic(sem_seg)
            else:
                ins_seg = ins_seg.where(ins_seg < tensors.PanopticMap.DIVISOR, 0)
                labels = tensors.PanopticMap.from_parts(
                    semantic=sem_seg, instance=ins_seg
                )
        else:
            labels = None
        if labels is not None:
            labels.translate_semantic_(info.translations_dataset, raises=True)
            labels.remove_instances_(list(info.background_ids))

        cap_data = CaptureData(
            times=times,
            images=multi_read(tensors.Image.read, "image", no_entries="error")(sources),
            segmentations=labels,
            depths=multi_read(tensors.DepthMap.read, "depth", no_entries="none")(
                sources
            ),
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
                    .as_subclass(tensors.PanopticMap)
                    .get_semantic_map()
                )
                for cat, fixed in info.depth_fixed.items():
                    cap_data.depths[i][sem_seg == cat] = fixed * info.depth_max

        return cap_data

    @classmethod
    def _load_motion_data(
        cls, sources: T.Sequence[MotionSources], info: Metadata
    ) -> MotionData:
        raise NotImplementedError(f"{cls.__name__} does not implement motion sources!")

    _data_cache: T.ClassVar[dict[str, InputData]] = {}

    @classmethod
    def _sequence_to_long(cls, sequence: str) -> torch.Tensor:
        """
        Convert the sequence string to a long tensor.
        """
        return torch.tensor(hash(sequence), dtype=torch.long)

    @classmethod
    @T.override
    def _load_data(cls, key: str, item: QueueItem, info: Metadata) -> InputData:
        from unipercept.data.tensors import PinholeCamera

        # Check for cache hit, should be a memmaped tensor
        # if key in cls._data_cache:
        #     return cls._data_cache[key].clone().contiguous()
        # types.utils.check_typeddict(item, QueueItem)
        # Captures
        caps_spec = item["captures"]
        caps_n = len(caps_spec)
        assert caps_n > 0

        try:
            caps_data = cls._load_capture_data(caps_spec, info)

            if "motions" in item:
                item_mots = item["motions"]
                item_mots_num = len(item_mots)
                assert item_mots_num > 0
                data_mots = cls._load_motion_data(item_mots, info)
            else:
                data_mots = None

            camera_spec = item["camera"]
            if not isinstance(camera_spec, T.Sequence):
                camera_spec = [camera_spec]
            camera_data = torch.stack(
                [
                    PinholeCamera.from_parameters(
                        focal_length=cam["focal_length"],
                        principal_point=cam["principal_point"],
                        translation=cam["translation"],
                        angles=cam["rotation"],
                        canvas=cam["image_size"],
                        convention=cam.get("convention", "opencv"),
                    )
                    for cam in camera_spec
                ]
            ).as_subclass(PinholeCamera)

            # IDs: (sequence, frame)
            ids = torch.tensor(
                [cls._sequence_to_long(item["sequence"]), item["frame"]],
                dtype=torch.long,
            )
        except Exception as e:
            msg = f"Failed to data at {key=}!\n\nItem: {pformat(item)}"
            raise RuntimeError(msg) from e

        input_data = InputData(
            ids=ids,
            captures=caps_data,
            motions=data_mots,
            cameras=camera_data,
            batch_size=[],
        )  # .memmap_()

        # cls._data_cache[key] = input_data

        return input_data  # .clone().contiguous()

    @classmethod
    @T.override
    def _check_manifest(cls, manifest: Manifest) -> bool:
        return cls.version == manifest["version"]

    def __call__(
        self, gatherer: T.Callable[[Manifest], tuple[str, QueueItem]] | None = None
    ) -> tuple[PerceptionDataqueue, PerceptionDatapipe]:
        """
        Combined
        """
        from unipercept.data.collect import ExtractIndividualFrames

        if queue_fn := getattr(self, "queue_fn", None) is not None:
            warnings.warn(
                "queue_fn is deprecated and will be removed in the future. "
                "Please provide the gatherer outside the dataset definition instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            gatherer = queue_fn
        if gatherer is None:
            gatherer = ExtractIndividualFrames()
        queue = self.build_queue(gatherer)
        pipe = self.build_pipe(queue)

        return queue, pipe

    if not T.TYPE_CHECKING:
        queue_fn: T.Callable[[Manifest], tuple[str, QueueItem]] | None = D.field(
            default=None,
            repr=False,
            metadata={
                "help": "Shorthand for defining a gatherer outside the loader (deprecated)."
            },
        )


class SubsampledDataset(PerceptionDataset, id=None):
    r"""
    A dataset that modifies a dataset by subsampling its manifest.
    """

    dataset: PerceptionDataset
    max_sequences: int | None = None
    max_captures: int | None = None
    max_motions: int | None = None

    def __post_init__(self, *args, **kwargs):
        if not any((self.max_sequences, self.max_captures, self.max_motions)):
            msg = (
                "No subsampling parameters provided! "
                f"Got: {self.max_sequences=}, {self.max_captures=}, {self.max_motions=}"
            )
            raise ValueError(msg)

        self.read_info = self.dataset.read_info

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    @property
    @T.override
    def version(self) -> str:
        return self.dataset.version

    @T.override
    def __repr__(self) -> str:
        atts = [
            f"{key}={val}"
            for key, val in {
                "max_sequences": self.max_sequences,
                "max_captures": self.max_captures,
                "max_motions": self.max_motions,
            }.items()
            if val is not None
        ]

        return f"{self.__class__.__name__}(dataset={repr(self.dataset)}, {', '.join(atts)})"

    @classmethod
    @T.override
    def options(cls) -> dict[str, T.Iterable[T.Any]]:
        msg = f"{cls.__name__} does not implement options!"
        raise NotImplementedError(msg)

    @T.override
    def _build_manifest(self) -> Manifest:
        mfst = self.dataset.manifest
        seqs = {}

        n_seq = self.max_sequences
        n_cap = self.max_captures
        n_mot = self.max_motions
        for key, seq in mfst["sequences"].items():
            if n_seq is not None:
                if n_seq <= 0:
                    break  # stop adding more sequences
                n_seq -= 1

            seq = copy.deepcopy(seq)
            if n_cap is not None and "captures" in seq:
                if n_cap <= 0:
                    break  # stop adding more captures
                caps = seq["captures"][:n_cap]
                n_cap -= len(caps)
                seq["captures"] = caps

            if n_mot is not None and "motions" in seq:
                if n_mot <= 0:
                    break
                mots = seq["motions"][:n_mot]
                n_mot -= len(mots)
                seq["motions"] = mots

            seqs[key] = seq

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": self.version,
            "sequences": seqs,
        }


class ConcatenatedDataset(PerceptionDataset, id=None):
    """
    A dataset that concatenates multiple datasets.

    The sequences in the manifest are tagged such that they can be identified as
    originating from different datasets in the concatenated dataset.
    This can then be used to route samples to the correct dataset for loading and
    processing.
    """

    datasets: T.Sequence[PerceptionDataset]

    def __post_init__(self, *args, **kwargs):
        self.datasets = sorted(self.datasets, key=lambda ds: repr(ds))

        if not all(
            ds.info.is_compatible(self.datasets[0].info) for ds in self.datasets
        ):
            msg = "Datasets have different metadata."
            raise ValueError(msg)

        self.read_info = self.datasets[0].read_info

    @property
    @T.override
    def version(self) -> str:
        return "+".join(ds.version for ds in self.datasets)

    @T.override
    def __repr__(self) -> str:
        ds_list = ",".join(repr(ds) for ds in self.datasets)
        return f"{self.__class__.__name__}(datasets={ds_list})"

    @classmethod
    @T.override
    def options(cls) -> dict[str, T.Iterable[T.Any]]:
        msg = f"{cls.__name__} does not implement options!"
        raise NotImplementedError(msg)

    @T.override
    def _build_manifest(self) -> Manifest:
        mfst_list = [ds.manifest for ds in self.datasets]

        version_list = [mfst["version"] for mfst in mfst_list]

        sequences: dict[str, ManifestSequence] = {}
        for mfst in mfst_list:
            # Check for duplicates
            if set(sequences.keys()) & set(mfst["sequences"].keys()):
                msg = f"Duplicate sequence names in datasets: {sequences.keys() & mfst['sequences'].keys()}"
                raise ValueError(msg)
            sequences.update(mfst["sequences"])

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "+".join(version_list),
            "sequences": sequences,
        }
