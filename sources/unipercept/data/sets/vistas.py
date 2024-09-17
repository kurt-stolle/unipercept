"""
Mapillary Vistas dataset
"""

from __future__ import annotations

import typing as T
from datetime import datetime
from typing import override

import typing_extensions as TX

from unipercept import file_io
from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.data.sets.cityscapes import CAMERA

from . import (
    RGB,
    CaptureRecord,
    CaptureSources,
    Manifest,
    ManifestSequence,
    Metadata,
    PerceptionDataset,
    SClass,
    SType,
)

CLASSES_AS_CITYSCAPES = [
    SClass(
        color=RGB(165, 42, 42),
        kind=SType.STUFF,
        dataset_id=0,
        unified_id=255,
        name="animal--bird",
    ),
    SClass(
        color=RGB(0, 192, 0),
        kind=SType.STUFF,
        dataset_id=1,
        unified_id=255,
        name="animal--ground-animal",
    ),
    SClass(
        color=RGB(196, 196, 196),
        kind=SType.STUFF,
        dataset_id=2,
        unified_id=1,
        name="construction--barrier--curb",
    ),
    SClass(
        color=RGB(190, 153, 153),
        kind=SType.STUFF,
        dataset_id=3,
        unified_id=4,
        name="construction--barrier--fence",
    ),
    SClass(
        color=RGB(180, 165, 180),
        kind=SType.STUFF,
        dataset_id=4,
        unified_id=255,
        name="construction--barrier--guard-rail",
    ),
    SClass(
        color=RGB(90, 120, 150),
        kind=SType.STUFF,
        dataset_id=5,
        unified_id=255,
        name="construction--barrier--other-barrier",
    ),
    SClass(
        color=RGB(102, 102, 156),
        kind=SType.STUFF,
        dataset_id=6,
        unified_id=3,
        name="construction--barrier--wall",
    ),
    SClass(
        color=RGB(128, 64, 6),
        kind=SType.STUFF,
        dataset_id=7,
        unified_id=0,
        name="construction--flat--bike-lane",
    ),
    SClass(
        color=RGB(140, 140, 200),
        kind=SType.STUFF,
        dataset_id=8,
        unified_id=0,
        name="construction--flat--crosswalk-plain",
    ),
    SClass(
        color=RGB(170, 170, 170),
        kind=SType.STUFF,
        dataset_id=9,
        unified_id=1,
        name="construction--flat--curb-cut",
    ),
    SClass(
        color=RGB(250, 170, 160),
        kind=SType.STUFF,
        dataset_id=10,
        unified_id=255,
        name="construction--flat--parking",
    ),
    SClass(
        color=RGB(96, 96, 96),
        kind=SType.STUFF,
        dataset_id=11,
        unified_id=1,
        name="construction--flat--pedestrian-area",
    ),
    SClass(
        color=RGB(230, 150, 140),
        kind=SType.STUFF,
        dataset_id=12,
        unified_id=255,
        name="construction--flat--rail-track",
    ),
    SClass(
        color=RGB(128, 64, 128),
        kind=SType.STUFF,
        dataset_id=13,
        unified_id=0,
        name="construction--flat--road",
    ),
    SClass(
        color=RGB(110, 110, 110),
        kind=SType.STUFF,
        dataset_id=14,
        unified_id=0,
        name="construction--flat--service-lane",
    ),
    SClass(
        color=RGB(244, 35, 232),
        kind=SType.STUFF,
        dataset_id=15,
        unified_id=1,
        name="construction--flat--sidewalk",
    ),
    SClass(
        color=RGB(150, 100, 100),
        kind=SType.STUFF,
        dataset_id=16,
        unified_id=255,
        name="construction--structure--bridge",
    ),
    SClass(
        color=RGB(70, 70, 70),
        kind=SType.STUFF,
        dataset_id=17,
        unified_id=2,
        name="construction--structure--building",
    ),
    SClass(
        color=RGB(150, 120, 90),
        kind=SType.STUFF,
        dataset_id=18,
        unified_id=255,
        name="construction--structure--tunnel",
    ),
    SClass(
        color=RGB(220, 20, 60),
        kind=SType.THING,
        dataset_id=19,
        unified_id=11,
        name="human--person",
    ),
    SClass(
        color=RGB(6, 0, 0),
        kind=SType.THING,
        dataset_id=20,
        unified_id=12,
        name="human--rider--bicyclist",
    ),
    SClass(
        color=RGB(6, 0, 100),
        kind=SType.THING,
        dataset_id=21,
        unified_id=12,
        name="human--rider--motorcyclist",
    ),
    SClass(
        color=RGB(6, 0, 200),
        kind=SType.THING,
        dataset_id=22,
        unified_id=12,
        name="human--rider--other-rider",
    ),
    SClass(
        color=RGB(200, 128, 128),
        kind=SType.STUFF,
        dataset_id=23,
        unified_id=0,
        name="marking--crosswalk-zebra",
    ),
    SClass(
        color=RGB(6, 6, 6),
        kind=SType.STUFF,
        dataset_id=24,
        unified_id=0,
        name="marking--general",
    ),
    SClass(
        color=RGB(64, 170, 64),
        kind=SType.STUFF,
        dataset_id=25,
        unified_id=9,
        name="nature--mountain",
    ),
    SClass(
        color=RGB(230, 160, 50),
        kind=SType.STUFF,
        dataset_id=26,
        unified_id=9,
        name="nature--sand",
    ),
    SClass(
        color=RGB(70, 130, 180),
        kind=SType.STUFF,
        dataset_id=27,
        unified_id=10,
        name="nature--sky",
    ),
    SClass(
        color=RGB(190, 6, 6),
        kind=SType.STUFF,
        dataset_id=28,
        unified_id=255,
        name="nature--snow",
    ),
    SClass(
        color=RGB(152, 251, 152),
        kind=SType.STUFF,
        dataset_id=29,
        unified_id=9,
        name="nature--terrain",
    ),
    SClass(
        color=RGB(107, 142, 35),
        kind=SType.STUFF,
        dataset_id=30,
        unified_id=8,
        name="nature--vegetation",
    ),
    SClass(
        color=RGB(0, 170, 130),
        kind=SType.STUFF,
        dataset_id=31,
        unified_id=255,
        name="nature--water",
    ),
    SClass(
        color=RGB(6, 6, 128),
        kind=SType.STUFF,
        dataset_id=32,
        unified_id=255,
        name="object--banner",
    ),
    SClass(
        color=RGB(250, 0, 30),
        kind=SType.STUFF,
        dataset_id=33,
        unified_id=255,
        name="object--bench",
    ),
    SClass(
        color=RGB(100, 140, 180),
        kind=SType.STUFF,
        dataset_id=34,
        unified_id=255,
        name="object--bike-rack",
    ),
    SClass(
        color=RGB(220, 220, 220),
        kind=SType.STUFF,
        dataset_id=35,
        unified_id=255,
        name="object--billboard",
    ),
    SClass(
        color=RGB(220, 128, 128),
        kind=SType.STUFF,
        dataset_id=36,
        unified_id=255,
        name="object--catch-basin",
    ),
    SClass(
        color=RGB(222, 40, 40),
        kind=SType.STUFF,
        dataset_id=37,
        unified_id=255,
        name="object--cctv-camera",
    ),
    SClass(
        color=RGB(100, 170, 30),
        kind=SType.STUFF,
        dataset_id=38,
        unified_id=255,
        name="object--fire-hydrant",
    ),
    SClass(
        color=RGB(40, 40, 40),
        kind=SType.STUFF,
        dataset_id=39,
        unified_id=255,
        name="object--junction-box",
    ),
    SClass(
        color=RGB(33, 33, 33),
        kind=SType.STUFF,
        dataset_id=40,
        unified_id=255,
        name="object--mailbox",
    ),
    SClass(
        color=RGB(100, 128, 160),
        kind=SType.STUFF,
        dataset_id=41,
        unified_id=0,
        name="object--manhole",
    ),
    SClass(
        color=RGB(142, 0, 0),
        kind=SType.STUFF,
        dataset_id=42,
        unified_id=255,
        name="object--phone-booth",
    ),
    SClass(
        color=RGB(70, 100, 150),
        kind=SType.STUFF,
        dataset_id=43,
        unified_id=0,
        name="object--pothole",
    ),
    SClass(
        color=RGB(210, 170, 100),
        kind=SType.STUFF,
        dataset_id=44,
        unified_id=5,
        name="object--street-light",
    ),
    SClass(
        color=RGB(153, 153, 153),
        kind=SType.STUFF,
        dataset_id=45,
        unified_id=5,
        name="object--support--pole",
    ),
    SClass(
        color=RGB(128, 128, 128),
        kind=SType.STUFF,
        dataset_id=46,
        unified_id=5,
        name="object--support--traffic-sign-frame",
    ),
    SClass(
        color=RGB(0, 0, 80),
        kind=SType.STUFF,
        dataset_id=47,
        unified_id=5,
        name="object--support--utility-pole",
    ),
    SClass(
        color=RGB(250, 170, 30),
        kind=SType.STUFF,
        dataset_id=48,
        unified_id=6,
        name="object--traffic-light",
    ),
    SClass(
        color=RGB(192, 192, 192),
        kind=SType.STUFF,
        dataset_id=49,
        unified_id=255,
        name="object--traffic-sign--back",
    ),
    SClass(
        color=RGB(220, 220, 0),
        kind=SType.STUFF,
        dataset_id=50,
        unified_id=7,
        name="object--traffic-sign--front",
    ),
    SClass(
        color=RGB(140, 140, 20),
        kind=SType.STUFF,
        dataset_id=51,
        unified_id=255,
        name="object--trash-can",
    ),
    SClass(
        color=RGB(119, 11, 32),
        kind=SType.THING,
        dataset_id=52,
        unified_id=18,
        name="object--vehicle--bicycle",
    ),
    SClass(
        color=RGB(150, 0, 6),
        kind=SType.STUFF,
        dataset_id=53,
        unified_id=255,
        name="object--vehicle--boat",
    ),
    SClass(
        color=RGB(0, 60, 100),
        kind=SType.THING,
        dataset_id=54,
        unified_id=15,
        name="object--vehicle--bus",
    ),
    SClass(
        color=RGB(0, 0, 142),
        kind=SType.THING,
        dataset_id=55,
        unified_id=13,
        name="object--vehicle--car",
    ),
    SClass(
        color=RGB(0, 0, 90),
        kind=SType.STUFF,
        dataset_id=56,
        unified_id=255,
        name="object--vehicle--caravan",
    ),  # Tim: class met id: 29, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 230),
        kind=SType.THING,
        dataset_id=57,
        unified_id=17,
        name="object--vehicle--motorcycle",
    ),
    SClass(
        color=RGB(0, 80, 100),
        kind=SType.THING,
        dataset_id=58,
        unified_id=16,
        name="object--vehicle--on-rails",
    ),
    SClass(
        color=RGB(128, 64, 64),
        kind=SType.STUFF,
        dataset_id=59,
        unified_id=255,
        name="object--vehicle--other-vehicle",
    ),
    SClass(
        color=RGB(0, 0, 110),
        kind=SType.STUFF,
        dataset_id=60,
        unified_id=255,
        name="object--vehicle--trailer",
    ),  # Tim: class met id: 30, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 70),
        kind=SType.THING,
        dataset_id=61,
        unified_id=14,
        name="object--vehicle--truck",
    ),
    SClass(
        color=RGB(0, 0, 192),
        kind=SType.STUFF,
        dataset_id=62,
        unified_id=255,
        name="object--vehicle--wheeled-slow",
    ),
    SClass(
        color=RGB(32, 32, 32),
        kind=SType.STUFF,
        dataset_id=63,
        unified_id=255,
        name="void--car-mount",
    ),
    SClass(
        color=RGB(120, 10, 10),
        kind=SType.STUFF,
        dataset_id=64,
        unified_id=255,
        name="void--ego-vehicle",
    ),  # Tim: class met id: 1, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 0),
        kind=SType.STUFF,
        dataset_id=65,
        unified_id=255,
        name="void--unlabeled",
    ),
]
CLASSES = [
    SClass(
        color=RGB(165, 42, 42),
        kind=SType.STUFF,
        dataset_id=0,
        unified_id=255,
        name="animal--bird",
    ),
    SClass(
        color=RGB(0, 192, 0),
        kind=SType.STUFF,
        dataset_id=1,
        unified_id=255,
        name="animal--ground-animal",
    ),
    SClass(
        color=RGB(196, 196, 196),
        kind=SType.STUFF,
        dataset_id=2,
        unified_id=1,
        name="construction--barrier--curb",
    ),
    SClass(
        color=RGB(190, 153, 153),
        kind=SType.STUFF,
        dataset_id=3,
        unified_id=4,
        name="construction--barrier--fence",
    ),
    SClass(
        color=RGB(180, 165, 180),
        kind=SType.STUFF,
        dataset_id=4,
        unified_id=255,
        name="construction--barrier--guard-rail",
    ),
    SClass(
        color=RGB(90, 120, 150),
        kind=SType.STUFF,
        dataset_id=5,
        unified_id=255,
        name="construction--barrier--other-barrier",
    ),
    SClass(
        color=RGB(102, 102, 156),
        kind=SType.STUFF,
        dataset_id=6,
        unified_id=3,
        name="construction--barrier--wall",
    ),
    SClass(
        color=RGB(128, 64, 6),
        kind=SType.STUFF,
        dataset_id=7,
        unified_id=0,
        name="construction--flat--bike-lane",
    ),
    SClass(
        color=RGB(140, 140, 200),
        kind=SType.STUFF,
        dataset_id=8,
        unified_id=0,
        name="construction--flat--crosswalk-plain",
    ),
    SClass(
        color=RGB(170, 170, 170),
        kind=SType.STUFF,
        dataset_id=9,
        unified_id=1,
        name="construction--flat--curb-cut",
    ),
    SClass(
        color=RGB(250, 170, 160),
        kind=SType.STUFF,
        dataset_id=10,
        unified_id=255,
        name="construction--flat--parking",
    ),
    SClass(
        color=RGB(96, 96, 96),
        kind=SType.STUFF,
        dataset_id=11,
        unified_id=1,
        name="construction--flat--pedestrian-area",
    ),
    SClass(
        color=RGB(230, 150, 140),
        kind=SType.STUFF,
        dataset_id=12,
        unified_id=255,
        name="construction--flat--rail-track",
    ),
    SClass(
        color=RGB(128, 64, 128),
        kind=SType.STUFF,
        dataset_id=13,
        unified_id=0,
        name="construction--flat--road",
    ),
    SClass(
        color=RGB(110, 110, 110),
        kind=SType.STUFF,
        dataset_id=14,
        unified_id=0,
        name="construction--flat--service-lane",
    ),
    SClass(
        color=RGB(244, 35, 232),
        kind=SType.STUFF,
        dataset_id=15,
        unified_id=1,
        name="construction--flat--sidewalk",
    ),
    SClass(
        color=RGB(150, 100, 100),
        kind=SType.STUFF,
        dataset_id=16,
        unified_id=255,
        name="construction--structure--bridge",
    ),
    SClass(
        color=RGB(70, 70, 70),
        kind=SType.STUFF,
        dataset_id=17,
        unified_id=2,
        name="construction--structure--building",
    ),
    SClass(
        color=RGB(150, 120, 90),
        kind=SType.STUFF,
        dataset_id=18,
        unified_id=255,
        name="construction--structure--tunnel",
    ),
    SClass(
        color=RGB(220, 20, 60),
        kind=SType.THING,
        dataset_id=19,
        unified_id=11,
        name="human--person",
    ),
    SClass(
        color=RGB(6, 0, 0),
        kind=SType.THING,
        dataset_id=20,
        unified_id=12,
        name="human--rider--bicyclist",
    ),
    SClass(
        color=RGB(6, 0, 100),
        kind=SType.THING,
        dataset_id=21,
        unified_id=12,
        name="human--rider--motorcyclist",
    ),
    SClass(
        color=RGB(6, 0, 200),
        kind=SType.THING,
        dataset_id=22,
        unified_id=12,
        name="human--rider--other-rider",
    ),
    SClass(
        color=RGB(200, 128, 128),
        kind=SType.STUFF,
        dataset_id=23,
        unified_id=0,
        name="marking--crosswalk-zebra",
    ),
    SClass(
        color=RGB(6, 6, 6),
        kind=SType.STUFF,
        dataset_id=24,
        unified_id=0,
        name="marking--general",
    ),
    SClass(
        color=RGB(64, 170, 64),
        kind=SType.STUFF,
        dataset_id=25,
        unified_id=9,
        name="nature--mountain",
    ),
    SClass(
        color=RGB(230, 160, 50),
        kind=SType.STUFF,
        dataset_id=26,
        unified_id=9,
        name="nature--sand",
    ),
    SClass(
        color=RGB(70, 130, 180),
        kind=SType.STUFF,
        dataset_id=27,
        unified_id=10,
        name="nature--sky",
    ),
    SClass(
        color=RGB(190, 6, 6),
        kind=SType.STUFF,
        dataset_id=28,
        unified_id=255,
        name="nature--snow",
    ),
    SClass(
        color=RGB(152, 251, 152),
        kind=SType.STUFF,
        dataset_id=29,
        unified_id=9,
        name="nature--terrain",
    ),
    SClass(
        color=RGB(107, 142, 35),
        kind=SType.STUFF,
        dataset_id=30,
        unified_id=8,
        name="nature--vegetation",
    ),
    SClass(
        color=RGB(0, 170, 130),
        kind=SType.STUFF,
        dataset_id=31,
        unified_id=255,
        name="nature--water",
    ),
    SClass(
        color=RGB(6, 6, 128),
        kind=SType.STUFF,
        dataset_id=32,
        unified_id=255,
        name="object--banner",
    ),
    SClass(
        color=RGB(250, 0, 30),
        kind=SType.STUFF,
        dataset_id=33,
        unified_id=255,
        name="object--bench",
    ),
    SClass(
        color=RGB(100, 140, 180),
        kind=SType.STUFF,
        dataset_id=34,
        unified_id=255,
        name="object--bike-rack",
    ),
    SClass(
        color=RGB(220, 220, 220),
        kind=SType.STUFF,
        dataset_id=35,
        unified_id=255,
        name="object--billboard",
    ),
    SClass(
        color=RGB(220, 128, 128),
        kind=SType.STUFF,
        dataset_id=36,
        unified_id=255,
        name="object--catch-basin",
    ),
    SClass(
        color=RGB(222, 40, 40),
        kind=SType.STUFF,
        dataset_id=37,
        unified_id=255,
        name="object--cctv-camera",
    ),
    SClass(
        color=RGB(100, 170, 30),
        kind=SType.STUFF,
        dataset_id=38,
        unified_id=255,
        name="object--fire-hydrant",
    ),
    SClass(
        color=RGB(40, 40, 40),
        kind=SType.STUFF,
        dataset_id=39,
        unified_id=255,
        name="object--junction-box",
    ),
    SClass(
        color=RGB(33, 33, 33),
        kind=SType.STUFF,
        dataset_id=40,
        unified_id=255,
        name="object--mailbox",
    ),
    SClass(
        color=RGB(100, 128, 160),
        kind=SType.STUFF,
        dataset_id=41,
        unified_id=0,
        name="object--manhole",
    ),
    SClass(
        color=RGB(142, 0, 0),
        kind=SType.STUFF,
        dataset_id=42,
        unified_id=255,
        name="object--phone-booth",
    ),
    SClass(
        color=RGB(70, 100, 150),
        kind=SType.STUFF,
        dataset_id=43,
        unified_id=0,
        name="object--pothole",
    ),
    SClass(
        color=RGB(210, 170, 100),
        kind=SType.STUFF,
        dataset_id=44,
        unified_id=5,
        name="object--street-light",
    ),
    SClass(
        color=RGB(153, 153, 153),
        kind=SType.STUFF,
        dataset_id=45,
        unified_id=5,
        name="object--support--pole",
    ),
    SClass(
        color=RGB(128, 128, 128),
        kind=SType.STUFF,
        dataset_id=46,
        unified_id=5,
        name="object--support--traffic-sign-frame",
    ),
    SClass(
        color=RGB(0, 0, 80),
        kind=SType.STUFF,
        dataset_id=47,
        unified_id=5,
        name="object--support--utility-pole",
    ),
    SClass(
        color=RGB(250, 170, 30),
        kind=SType.STUFF,
        dataset_id=48,
        unified_id=6,
        name="object--traffic-light",
    ),
    SClass(
        color=RGB(192, 192, 192),
        kind=SType.STUFF,
        dataset_id=49,
        unified_id=255,
        name="object--traffic-sign--back",
    ),
    SClass(
        color=RGB(220, 220, 0),
        kind=SType.STUFF,
        dataset_id=50,
        unified_id=7,
        name="object--traffic-sign--front",
    ),
    SClass(
        color=RGB(140, 140, 20),
        kind=SType.STUFF,
        dataset_id=51,
        unified_id=255,
        name="object--trash-can",
    ),
    SClass(
        color=RGB(119, 11, 32),
        kind=SType.THING,
        dataset_id=52,
        unified_id=18,
        name="object--vehicle--bicycle",
    ),
    SClass(
        color=RGB(150, 0, 6),
        kind=SType.STUFF,
        dataset_id=53,
        unified_id=255,
        name="object--vehicle--boat",
    ),
    SClass(
        color=RGB(0, 60, 100),
        kind=SType.THING,
        dataset_id=54,
        unified_id=15,
        name="object--vehicle--bus",
    ),
    SClass(
        color=RGB(0, 0, 142),
        kind=SType.THING,
        dataset_id=55,
        unified_id=13,
        name="object--vehicle--car",
    ),
    SClass(
        color=RGB(0, 0, 90),
        kind=SType.STUFF,
        dataset_id=56,
        unified_id=255,
        name="object--vehicle--caravan",
    ),  # Tim: class met id: 29, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 230),
        kind=SType.THING,
        dataset_id=57,
        unified_id=17,
        name="object--vehicle--motorcycle",
    ),
    SClass(
        color=RGB(0, 80, 100),
        kind=SType.THING,
        dataset_id=58,
        unified_id=16,
        name="object--vehicle--on-rails",
    ),
    SClass(
        color=RGB(128, 64, 64),
        kind=SType.STUFF,
        dataset_id=59,
        unified_id=255,
        name="object--vehicle--other-vehicle",
    ),
    SClass(
        color=RGB(0, 0, 110),
        kind=SType.STUFF,
        dataset_id=60,
        unified_id=255,
        name="object--vehicle--trailer",
    ),  # Tim: class met id: 30, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 70),
        kind=SType.THING,
        dataset_id=61,
        unified_id=14,
        name="object--vehicle--truck",
    ),
    SClass(
        color=RGB(0, 0, 192),
        kind=SType.STUFF,
        dataset_id=62,
        unified_id=255,
        name="object--vehicle--wheeled-slow",
    ),
    SClass(
        color=RGB(32, 32, 32),
        kind=SType.STUFF,
        dataset_id=63,
        unified_id=255,
        name="void--car-mount",
    ),
    SClass(
        color=RGB(120, 10, 10),
        kind=SType.STUFF,
        dataset_id=64,
        unified_id=255,
        name="void--ego-vehicle",
    ),  # Tim: class met id: 1, staat niet in cityscapes config
    SClass(
        color=RGB(0, 0, 0),
        kind=SType.STUFF,
        dataset_id=65,
        unified_id=255,
        name="void--unlabeled",
    ),
]


def get_info(variant: str = ""):
    match variant:
        case "cityscapes":
            use_cityscapes = True
        case "":
            use_cityscapes = False
        case _:
            raise ValueError(f"Unknown variant {variant!r}")
    return Metadata.from_parameters(
        CLASSES if not use_cityscapes else CLASSES_AS_CITYSCAPES,
        depth_max=80.0,
        fps=17.0,
    )


class MapillaryVistasDataset(PerceptionDataset, info=get_info, id="mapillary-vistas"):
    """
    Vistas dataset using the provided labeling scheme.

    The dataset consists of single images, i.e. there are no sequences.
    """

    split: T.Literal["train", "val", "test"]
    root: str = "//datasets/vistas"

    @classmethod
    @TX.override
    def options(cls):
        return {
            "split": ["train", "val", "test"],
        }

    @override
    def _build_manifest(self) -> Manifest:
        from tqdm import tqdm

        split_dir = file_io.Path(self.root) / f"{self.split}ing"
        with PseudoGenerator() as pseudogen:
            sequences: T.Mapping[str, ManifestSequence] = {}

            image_list = list((split_dir / "images").glob("*.jpg"))
            image_list.sort(key=lambda p: p.stem)

            if len(image_list) == 0:
                raise RuntimeError(f"No images found in {split_dir}")

            for image_path in tqdm(image_list, desc="Building manifest", unit="image"):
                cap_key = image_path.stem
                seq_key = cap_key  # NOTE: no sequences

                sources: CaptureSources = {
                    "image": {
                        "path": str(image_path),
                    },
                    "panoptic": {
                        "path": str(split_dir / "v2.0" / "panoptic" / f"{cap_key}.png"),
                        "meta": {"format": "vistas"},
                    },
                    "depth": {
                        "path": str(split_dir / "v2.0" / "depth" / f"{cap_key}.tiff"),
                        "meta": {"format": "tiff"},
                    },
                }

                def _source_exists(src):
                    return file_io.Path(src["path"]).exists()

                if not _source_exists(sources["panoptic"]) or self.split == "test":
                    del sources["panoptic"]
                if not _source_exists(sources["depth"]) or self.split == "test":
                    pseudogen.add_depth_generator_task(
                        sources["image"]["path"], sources["depth"]["path"]
                    )

                camera = (
                    CAMERA.to_canonical()
                )  # TODO: fill with estimated camera parameters
                captures: list[CaptureRecord] = [
                    {"primary_key": cap_key, "sources": sources}
                ]

                # Create sequence item
                seq_item: ManifestSequence = {
                    "camera": camera,
                    "fps": 15,
                    "captures": captures,
                }
                sequences[seq_key] = seq_item

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
            "sequences": sequences,
        }
