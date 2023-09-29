"""
Mapillary Vistas dataset
"""

import functools
import re
from enum import Enum
from pathlib import Path
from typing import Iterable, NamedTuple

from unipercept.data.structures import FileResource

from ._meta import Metadata, generate_metadata
from ._pipes import get_image_size
from .structures import Category, Record
from .Wrapper import DatasetWrapper


class Mapillary(DatasetWrapper, split_names={"training", "validation", "testing"}):
    """
    Mapillary dataset mapped to Cityscapes dataset classes.
    """

    @classmethod
    def get_root(cls) -> Path:
        return super().get_root() / "mapillary"

    def discover(self):
        root_path = self.get_root()

        images_path = root_path / self.split.value / "images"
        assert images_path.is_dir(), f"Not a directory: {images_path}"

        panoptics_path = root_path / self.split.value / "instances"
        assert panoptics_path.is_dir(), f"Not a directory: {panoptics_path}"

        # Load samples
        for image_path in images_path.iterdir():
            width, height = self.read_shape(image_path)
            image_id = image_path.name[:-4]
            pans_image_path = image_id + ".png"
            record = Record(
                image_id=image_id,
                image=FileResource(path=image_path.as_posix(), type="image"),
                height=height,
                width=width,
            )

            try:
                record["panseg"] = FileResource(path=(panoptics_path / pans_image_path).as_posix(), type="vistas")
            except KeyError:
                pass

            yield record

    @classmethod
    def is_installed(cls):
        return cls.get_root().is_dir()

    @functools.cached_property
    def metadata(self) -> Metadata:
        categories_cs = [
            {"color": (165, 42, 42), "isthing": 0, "id": 0, "trainId": 255, "name": "animal--bird"},
            {"color": (0, 192, 0), "isthing": 0, "id": 1, "trainId": 255, "name": "animal--ground-animal"},
            {"color": (196, 196, 196), "isthing": 0, "id": 2, "trainId": 1, "name": "construction--barrier--curb"},
            {"color": (190, 153, 153), "isthing": 0, "id": 3, "trainId": 4, "name": "construction--barrier--fence"},
            {
                "color": (180, 165, 180),
                "isthing": 0,
                "id": 4,
                "trainId": 255,
                "name": "construction--barrier--guard-rail",
            },
            {
                "color": (90, 120, 150),
                "isthing": 0,
                "id": 5,
                "trainId": 255,
                "name": "construction--barrier--other-barrier",
            },
            {"color": (102, 102, 156), "isthing": 0, "id": 6, "trainId": 3, "name": "construction--barrier--wall"},
            {"color": (128, 64, 6), "isthing": 0, "id": 7, "trainId": 0, "name": "construction--flat--bike-lane"},
            {
                "color": (140, 140, 200),
                "isthing": 0,
                "id": 8,
                "trainId": 0,
                "name": "construction--flat--crosswalk-plain",
            },
            {"color": (170, 170, 170), "isthing": 0, "id": 9, "trainId": 1, "name": "construction--flat--curb-cut"},
            {"color": (250, 170, 160), "isthing": 0, "id": 10, "trainId": 255, "name": "construction--flat--parking"},
            {
                "color": (96, 96, 96),
                "isthing": 0,
                "id": 11,
                "trainId": 1,
                "name": "construction--flat--pedestrian-area",
            },
            {
                "color": (230, 150, 140),
                "isthing": 0,
                "id": 12,
                "trainId": 255,
                "name": "construction--flat--rail-track",
            },
            {"color": (128, 64, 128), "isthing": 0, "id": 13, "trainId": 0, "name": "construction--flat--road"},
            {
                "color": (110, 110, 110),
                "isthing": 0,
                "id": 14,
                "trainId": 0,
                "name": "construction--flat--service-lane",
            },
            {"color": (244, 35, 232), "isthing": 0, "id": 15, "trainId": 1, "name": "construction--flat--sidewalk"},
            {
                "color": (150, 100, 100),
                "isthing": 0,
                "id": 16,
                "trainId": 255,
                "name": "construction--structure--bridge",
            },
            {"color": (70, 70, 70), "isthing": 0, "id": 17, "trainId": 2, "name": "construction--structure--building"},
            {
                "color": (150, 120, 90),
                "isthing": 0,
                "id": 18,
                "trainId": 255,
                "name": "construction--structure--tunnel",
            },
            {"color": (220, 20, 60), "isthing": 1, "id": 19, "trainId": 11, "name": "human--person"},
            {"color": (6, 0, 0), "isthing": 1, "id": 20, "trainId": 12, "name": "human--rider--bicyclist"},
            {"color": (6, 0, 100), "isthing": 1, "id": 21, "trainId": 12, "name": "human--rider--motorcyclist"},
            {"color": (6, 0, 200), "isthing": 1, "id": 22, "trainId": 12, "name": "human--rider--other-rider"},
            {"color": (200, 128, 128), "isthing": 0, "id": 23, "trainId": 0, "name": "marking--crosswalk-zebra"},
            {"color": (6, 6, 6), "isthing": 0, "id": 24, "trainId": 0, "name": "marking--general"},
            {"color": (64, 170, 64), "isthing": 0, "id": 25, "trainId": 9, "name": "nature--mountain"},
            {"color": (230, 160, 50), "isthing": 0, "id": 26, "trainId": 9, "name": "nature--sand"},
            {"color": (70, 130, 180), "isthing": 0, "id": 27, "trainId": 10, "name": "nature--sky"},
            {"color": (190, 6, 6), "isthing": 0, "id": 28, "trainId": 255, "name": "nature--snow"},
            {"color": (152, 251, 152), "isthing": 0, "id": 29, "trainId": 9, "name": "nature--terrain"},
            {"color": (107, 142, 35), "isthing": 0, "id": 30, "trainId": 8, "name": "nature--vegetation"},
            {"color": (0, 170, 130), "isthing": 0, "id": 31, "trainId": 255, "name": "nature--water"},
            {"color": (6, 6, 128), "isthing": 0, "id": 32, "trainId": 255, "name": "object--banner"},
            {"color": (250, 0, 30), "isthing": 0, "id": 33, "trainId": 255, "name": "object--bench"},
            {"color": (100, 140, 180), "isthing": 0, "id": 34, "trainId": 255, "name": "object--bike-rack"},
            {"color": (220, 220, 220), "isthing": 0, "id": 35, "trainId": 255, "name": "object--billboard"},
            {"color": (220, 128, 128), "isthing": 0, "id": 36, "trainId": 255, "name": "object--catch-basin"},
            {"color": (222, 40, 40), "isthing": 0, "id": 37, "trainId": 255, "name": "object--cctv-camera"},
            {"color": (100, 170, 30), "isthing": 0, "id": 38, "trainId": 255, "name": "object--fire-hydrant"},
            {"color": (40, 40, 40), "isthing": 0, "id": 39, "trainId": 255, "name": "object--junction-box"},
            {"color": (33, 33, 33), "isthing": 0, "id": 40, "trainId": 255, "name": "object--mailbox"},
            {"color": (100, 128, 160), "isthing": 0, "id": 41, "trainId": 0, "name": "object--manhole"},
            {"color": (142, 0, 0), "isthing": 0, "id": 42, "trainId": 255, "name": "object--phone-booth"},
            {"color": (70, 100, 150), "isthing": 0, "id": 43, "trainId": 0, "name": "object--pothole"},
            {"color": (210, 170, 100), "isthing": 0, "id": 44, "trainId": 5, "name": "object--street-light"},
            {"color": (153, 153, 153), "isthing": 0, "id": 45, "trainId": 5, "name": "object--support--pole"},
            {
                "color": (128, 128, 128),
                "isthing": 0,
                "id": 46,
                "trainId": 5,
                "name": "object--support--traffic-sign-frame",
            },
            {"color": (0, 0, 80), "isthing": 0, "id": 47, "trainId": 5, "name": "object--support--utility-pole"},
            {"color": (250, 170, 30), "isthing": 0, "id": 48, "trainId": 6, "name": "object--traffic-light"},
            {"color": (192, 192, 192), "isthing": 0, "id": 49, "trainId": 255, "name": "object--traffic-sign--back"},
            {"color": (220, 220, 0), "isthing": 0, "id": 50, "trainId": 7, "name": "object--traffic-sign--front"},
            {"color": (140, 140, 20), "isthing": 0, "id": 51, "trainId": 255, "name": "object--trash-can"},
            {"color": (119, 11, 32), "isthing": 1, "id": 52, "trainId": 18, "name": "object--vehicle--bicycle"},
            {"color": (150, 0, 6), "isthing": 0, "id": 53, "trainId": 255, "name": "object--vehicle--boat"},
            {"color": (0, 60, 100), "isthing": 1, "id": 54, "trainId": 15, "name": "object--vehicle--bus"},
            {"color": (0, 0, 142), "isthing": 1, "id": 55, "trainId": 13, "name": "object--vehicle--car"},
            {
                "color": (0, 0, 90),
                "isthing": 0,
                "id": 56,
                "trainId": 255,
                "name": "object--vehicle--caravan",
            },  # Tim: class met id: 29, staat niet in cityscapes config
            {"color": (0, 0, 230), "isthing": 1, "id": 57, "trainId": 17, "name": "object--vehicle--motorcycle"},
            {"color": (0, 80, 100), "isthing": 1, "id": 58, "trainId": 16, "name": "object--vehicle--on-rails"},
            {"color": (128, 64, 64), "isthing": 0, "id": 59, "trainId": 255, "name": "object--vehicle--other-vehicle"},
            {
                "color": (0, 0, 110),
                "isthing": 0,
                "id": 60,
                "trainId": 255,
                "name": "object--vehicle--trailer",
            },  # Tim: class met id: 30, staat niet in cityscapes config
            {"color": (0, 0, 70), "isthing": 1, "id": 61, "trainId": 14, "name": "object--vehicle--truck"},
            {"color": (0, 0, 192), "isthing": 0, "id": 62, "trainId": 255, "name": "object--vehicle--wheeled-slow"},
            {"color": (32, 32, 32), "isthing": 0, "id": 63, "trainId": 255, "name": "void--car-mount"},
            {
                "color": (120, 10, 10),
                "isthing": 0,
                "id": 64,
                "trainId": 255,
                "name": "void--ego-vehicle",
            },  # Tim: class met id: 1, staat niet in cityscapes config
            {"color": (0, 0, 0), "isthing": 0, "id": 65, "trainId": 255, "name": "void--unlabeled"},
        ]
        categories = [
            {"color": (165, 42, 42), "isthing": 0, "id": 0, "trainId": 255, "name": "animal--bird"},
            {"color": (0, 192, 0), "isthing": 0, "id": 1, "trainId": 255, "name": "animal--ground-animal"},
            {"color": (196, 196, 196), "isthing": 0, "id": 2, "trainId": 1, "name": "construction--barrier--curb"},
            {"color": (190, 153, 153), "isthing": 0, "id": 3, "trainId": 4, "name": "construction--barrier--fence"},
            {
                "color": (180, 165, 180),
                "isthing": 0,
                "id": 4,
                "trainId": 255,
                "name": "construction--barrier--guard-rail",
            },
            {
                "color": (90, 120, 150),
                "isthing": 0,
                "id": 5,
                "trainId": 255,
                "name": "construction--barrier--other-barrier",
            },
            {"color": (102, 102, 156), "isthing": 0, "id": 6, "trainId": 3, "name": "construction--barrier--wall"},
            {"color": (128, 64, 6), "isthing": 0, "id": 7, "trainId": 0, "name": "construction--flat--bike-lane"},
            {
                "color": (140, 140, 200),
                "isthing": 0,
                "id": 8,
                "trainId": 0,
                "name": "construction--flat--crosswalk-plain",
            },
            {"color": (170, 170, 170), "isthing": 0, "id": 9, "trainId": 1, "name": "construction--flat--curb-cut"},
            {"color": (250, 170, 160), "isthing": 0, "id": 10, "trainId": 255, "name": "construction--flat--parking"},
            {
                "color": (96, 96, 96),
                "isthing": 0,
                "id": 11,
                "trainId": 1,
                "name": "construction--flat--pedestrian-area",
            },
            {
                "color": (230, 150, 140),
                "isthing": 0,
                "id": 12,
                "trainId": 255,
                "name": "construction--flat--rail-track",
            },
            {"color": (128, 64, 128), "isthing": 0, "id": 13, "trainId": 0, "name": "construction--flat--road"},
            {
                "color": (110, 110, 110),
                "isthing": 0,
                "id": 14,
                "trainId": 0,
                "name": "construction--flat--service-lane",
            },
            {"color": (244, 35, 232), "isthing": 0, "id": 15, "trainId": 1, "name": "construction--flat--sidewalk"},
            {
                "color": (150, 100, 100),
                "isthing": 0,
                "id": 16,
                "trainId": 255,
                "name": "construction--structure--bridge",
            },
            {"color": (70, 70, 70), "isthing": 0, "id": 17, "trainId": 2, "name": "construction--structure--building"},
            {
                "color": (150, 120, 90),
                "isthing": 0,
                "id": 18,
                "trainId": 255,
                "name": "construction--structure--tunnel",
            },
            {"color": (220, 20, 60), "isthing": 1, "id": 19, "trainId": 11, "name": "human--person"},
            {"color": (6, 0, 0), "isthing": 1, "id": 20, "trainId": 12, "name": "human--rider--bicyclist"},
            {"color": (6, 0, 100), "isthing": 1, "id": 21, "trainId": 12, "name": "human--rider--motorcyclist"},
            {"color": (6, 0, 200), "isthing": 1, "id": 22, "trainId": 12, "name": "human--rider--other-rider"},
            {"color": (200, 128, 128), "isthing": 0, "id": 23, "trainId": 0, "name": "marking--crosswalk-zebra"},
            {"color": (6, 6, 6), "isthing": 0, "id": 24, "trainId": 0, "name": "marking--general"},
            {"color": (64, 170, 64), "isthing": 0, "id": 25, "trainId": 9, "name": "nature--mountain"},
            {"color": (230, 160, 50), "isthing": 0, "id": 26, "trainId": 9, "name": "nature--sand"},
            {"color": (70, 130, 180), "isthing": 0, "id": 27, "trainId": 10, "name": "nature--sky"},
            {"color": (190, 6, 6), "isthing": 0, "id": 28, "trainId": 255, "name": "nature--snow"},
            {"color": (152, 251, 152), "isthing": 0, "id": 29, "trainId": 9, "name": "nature--terrain"},
            {"color": (107, 142, 35), "isthing": 0, "id": 30, "trainId": 8, "name": "nature--vegetation"},
            {"color": (0, 170, 130), "isthing": 0, "id": 31, "trainId": 255, "name": "nature--water"},
            {"color": (6, 6, 128), "isthing": 0, "id": 32, "trainId": 255, "name": "object--banner"},
            {"color": (250, 0, 30), "isthing": 0, "id": 33, "trainId": 255, "name": "object--bench"},
            {"color": (100, 140, 180), "isthing": 0, "id": 34, "trainId": 255, "name": "object--bike-rack"},
            {"color": (220, 220, 220), "isthing": 0, "id": 35, "trainId": 255, "name": "object--billboard"},
            {"color": (220, 128, 128), "isthing": 0, "id": 36, "trainId": 255, "name": "object--catch-basin"},
            {"color": (222, 40, 40), "isthing": 0, "id": 37, "trainId": 255, "name": "object--cctv-camera"},
            {"color": (100, 170, 30), "isthing": 0, "id": 38, "trainId": 255, "name": "object--fire-hydrant"},
            {"color": (40, 40, 40), "isthing": 0, "id": 39, "trainId": 255, "name": "object--junction-box"},
            {"color": (33, 33, 33), "isthing": 0, "id": 40, "trainId": 255, "name": "object--mailbox"},
            {"color": (100, 128, 160), "isthing": 0, "id": 41, "trainId": 0, "name": "object--manhole"},
            {"color": (142, 0, 0), "isthing": 0, "id": 42, "trainId": 255, "name": "object--phone-booth"},
            {"color": (70, 100, 150), "isthing": 0, "id": 43, "trainId": 0, "name": "object--pothole"},
            {"color": (210, 170, 100), "isthing": 0, "id": 44, "trainId": 5, "name": "object--street-light"},
            {"color": (153, 153, 153), "isthing": 0, "id": 45, "trainId": 5, "name": "object--support--pole"},
            {
                "color": (128, 128, 128),
                "isthing": 0,
                "id": 46,
                "trainId": 5,
                "name": "object--support--traffic-sign-frame",
            },
            {"color": (0, 0, 80), "isthing": 0, "id": 47, "trainId": 5, "name": "object--support--utility-pole"},
            {"color": (250, 170, 30), "isthing": 0, "id": 48, "trainId": 6, "name": "object--traffic-light"},
            {"color": (192, 192, 192), "isthing": 0, "id": 49, "trainId": 255, "name": "object--traffic-sign--back"},
            {"color": (220, 220, 0), "isthing": 0, "id": 50, "trainId": 7, "name": "object--traffic-sign--front"},
            {"color": (140, 140, 20), "isthing": 0, "id": 51, "trainId": 255, "name": "object--trash-can"},
            {"color": (119, 11, 32), "isthing": 1, "id": 52, "trainId": 18, "name": "object--vehicle--bicycle"},
            {"color": (150, 0, 6), "isthing": 0, "id": 53, "trainId": 255, "name": "object--vehicle--boat"},
            {"color": (0, 60, 100), "isthing": 1, "id": 54, "trainId": 15, "name": "object--vehicle--bus"},
            {"color": (0, 0, 142), "isthing": 1, "id": 55, "trainId": 13, "name": "object--vehicle--car"},
            {
                "color": (0, 0, 90),
                "isthing": 0,
                "id": 56,
                "trainId": 255,
                "name": "object--vehicle--caravan",
            },  # Tim: class met id: 29, staat niet in cityscapes config
            {"color": (0, 0, 230), "isthing": 1, "id": 57, "trainId": 17, "name": "object--vehicle--motorcycle"},
            {"color": (0, 80, 100), "isthing": 1, "id": 58, "trainId": 16, "name": "object--vehicle--on-rails"},
            {"color": (128, 64, 64), "isthing": 0, "id": 59, "trainId": 255, "name": "object--vehicle--other-vehicle"},
            {
                "color": (0, 0, 110),
                "isthing": 0,
                "id": 60,
                "trainId": 255,
                "name": "object--vehicle--trailer",
            },  # Tim: class met id: 30, staat niet in cityscapes config
            {"color": (0, 0, 70), "isthing": 1, "id": 61, "trainId": 14, "name": "object--vehicle--truck"},
            {"color": (0, 0, 192), "isthing": 0, "id": 62, "trainId": 255, "name": "object--vehicle--wheeled-slow"},
            {"color": (32, 32, 32), "isthing": 0, "id": 63, "trainId": 255, "name": "void--car-mount"},
            {
                "color": (120, 10, 10),
                "isthing": 0,
                "id": 64,
                "trainId": 255,
                "name": "void--ego-vehicle",
            },  # Tim: class met id: 1, staat niet in cityscapes config
            {"color": (0, 0, 0), "isthing": 0, "id": 65, "trainId": 255, "name": "void--unlabeled"},
        ]
        return generate_metadata(
            categories=categories,  # type: ignore
            depth_max=80.0,
            label_divisor=256,
        )
