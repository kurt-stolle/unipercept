"""
WildDash dataset.
"""

from __future__ import annotations

import json
import typing as T
from datetime import datetime

from unipercept import file_io
from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.data.sets._base import PerceptionDataset, create_metadata

if T.TYPE_CHECKING:
    import unipercept as up


def get_info():
    categories = [
        {
            "color": (0, 0, 0),
            "isthing": 0,
            "id": 0,
            "trainId": 255,
            "name": "unlabeled(void)",
        },
        {
            "color": (0, 20, 50),
            "isthing": 1,
            "id": 1,
            "trainId": 0,
            "name": "ego vehicle(vehicle)",
        },
        {
            "color": (20, 20, 20),
            "isthing": 0,
            "id": 2,
            "trainId": 255,
            "name": "rectification border(void)",
        },
        {
            "color": (0, 0, 0),
            "isthing": 0,
            "id": 3,
            "trainId": 255,
            "name": "out of roi(void)",
        },
        {
            "color": (0, 0, 0),
            "isthing": 0,
            "id": 4,
            "trainId": 255,
            "name": "static(void)",
        },
        {
            "color": (111, 74, 0),
            "isthing": 0,
            "id": 5,
            "trainId": 255,
            "name": "dynamic(void)",
        },
        {
            "color": (81, 0, 81),
            "isthing": 0,
            "id": 6,
            "trainId": 255,
            "name": "ground(flat)",
        },
        {
            "color": (128, 64, 128),
            "isthing": 0,
            "id": 7,
            "trainId": 1,
            "name": "road(flat)",
        },
        {
            "color": (244, 35, 232),
            "isthing": 0,
            "id": 8,
            "trainId": 2,
            "name": "sidewalk(flat)",
        },
        {
            "color": (250, 170, 160),
            "isthing": 0,
            "id": 9,
            "trainId": 255,
            "name": "parking(flat)",
        },
        {
            "color": (230, 150, 140),
            "isthing": 0,
            "id": 10,
            "trainId": 255,
            "name": "rail track(flat)",
        },
        {
            "color": (70, 70, 70),
            "isthing": 0,
            "id": 11,
            "trainId": 3,
            "name": "building(construction)",
        },
        {
            "color": (102, 102, 156),
            "isthing": 0,
            "id": 12,
            "trainId": 4,
            "name": "wall(construction)",
        },
        {
            "color": (190, 153, 153),
            "isthing": 0,
            "id": 13,
            "trainId": 5,
            "name": "fence(construction)",
        },
        {
            "color": (180, 165, 180),
            "isthing": 0,
            "id": 14,
            "trainId": 6,
            "name": "guard rail(construction)",
        },
        {
            "color": (150, 100, 100),
            "isthing": 0,
            "id": 15,
            "trainId": 255,
            "name": "bridge(construction)",
        },
        {
            "color": (150, 120, 90),
            "isthing": 0,
            "id": 16,
            "trainId": 255,
            "name": "tunnel(construction)",
        },
        {
            "color": (153, 153, 153),
            "isthing": 0,
            "id": 17,
            "trainId": 7,
            "name": "pole(object)",
        },
        {
            "color": (153, 153, 153),
            "isthing": 0,
            "id": 18,
            "trainId": 255,
            "name": "polegroup(object)",
        },
        {
            "color": (250, 170, 30),
            "isthing": 0,
            "id": 19,
            "trainId": 8,
            "name": "traffic light(object)",
        },
        {
            "color": (220, 220, 0),
            "isthing": 0,
            "id": 20,
            "trainId": 9,
            "name": "traffic sign(object)",
        },
        {
            "color": (107, 142, 35),
            "isthing": 0,
            "id": 21,
            "trainId": 10,
            "name": "vegetation(nature)",
        },
        {
            "color": (152, 251, 152),
            "isthing": 0,
            "id": 22,
            "trainId": 11,
            "name": "terrain(nature)",
        },
        {
            "color": (70, 130, 180),
            "isthing": 0,
            "id": 23,
            "trainId": 12,
            "name": "sky(sky)",
        },
        {
            "color": (220, 20, 60),
            "isthing": 1,
            "id": 24,
            "trainId": 13,
            "name": "person(human)",
        },
        {
            "color": (255, 0, 0),
            "isthing": 1,
            "id": 25,
            "trainId": 14,
            "name": "rider(human)",
        },
        {
            "color": (0, 0, 142),
            "isthing": 1,
            "id": 26,
            "trainId": 15,
            "name": "car(vehicle)",
        },
        {
            "color": (0, 0, 70),
            "isthing": 1,
            "id": 27,
            "trainId": 16,
            "name": "truck(vehicle)",
        },
        {
            "color": (0, 60, 100),
            "isthing": 1,
            "id": 28,
            "trainId": 17,
            "name": "bus(vehicle)",
        },
        {
            "color": (0, 0, 90),
            "isthing": 1,
            "id": 29,
            "trainId": 255,
            "name": "caravan(vehicle)",
        },
        {
            "color": (0, 0, 110),
            "isthing": 1,
            "id": 30,
            "trainId": 255,
            "name": "trailer(vehicle)",
        },
        {
            "color": (0, 80, 100),
            "isthing": 1,
            "id": 31,
            "trainId": 255,
            "name": "trains and trams(vehicle)",
        },
        {
            "color": (0, 0, 230),
            "isthing": 1,
            "id": 32,
            "trainId": 18,
            "name": "motorcycle(vehicle)",
        },
        {
            "color": (119, 11, 32),
            "isthing": 1,
            "id": 33,
            "trainId": 19,
            "name": "bicycle(vehicle)",
        },
        {
            "color": (40, 0, 100),
            "isthing": 1,
            "id": 34,
            "trainId": 20,
            "name": "pickup-truck(vehicle)",
        },
        {
            "color": (0, 40, 120),
            "isthing": 1,
            "id": 35,
            "trainId": 21,
            "name": "van(vehicle)",
        },
        {
            "color": (174, 64, 67),
            "isthing": 0,
            "id": 36,
            "trainId": 22,
            "name": "billboard(object)",
        },
        {
            "color": (210, 170, 100),
            "isthing": 0,
            "id": 37,
            "trainId": 23,
            "name": "street-light(object)",
        },
        {
            "color": (196, 176, 128),
            "isthing": 0,
            "id": 38,
            "trainId": 24,
            "name": "road-marking(flat)",
        },
    ]
    return create_metadata(
        categories=categories, depth_max=80.0, label_divisor=int(1e8)
    )


class WildDashDataset(PerceptionDataset, id="wilddash", info=get_info):
    """WildDash dataset."""

    root = "//datasets/wilddash"
    split: T.Literal["train", "val"]

    def _build_manifest(self) -> up.data.types.Manifest:
        pan_path = file_io.Path(self.root) / "panoptic.json"
        with open(pan_path) as fh:
            pan: up.data.types.COCOManifest = json.load(fh)

        ann_map = {ann["file_name"]: ann["segments_info"] for ann in pan["annotations"]}

        root_path = file_io.Path(self.root)

        src_image = root_path / self.split / "images"
        src_panseg = root_path / "panoptic"
        for src in (src_image, src_panseg):
            assert src.is_dir(), f"Not a directory: {src!s}"

        sequences: dict[str, up.data.types.ManifestSequence] = {}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
            "sequences": sequences,
        }
        # Load samples
        for image_path in src_image.iterdir():
            width, height = self.read_shape(image_path)
            image_id = image_path.name.split(".", maxsplit=1)[0]
            pans_image_path = image_id + ".png"
            record = Record(
                image_id=image_id,
                image=FileResource(path=image_path.as_posix(), type="image"),
                height=height,
                width=width,
            )

            try:
                record["annotations"] = self.annotations[pans_image_path]
            except KeyError:
                pass

            try:
                record["panseg"] = FileResource(
                    path=(src_panseg / pans_image_path).as_posix(), type="wilddash"
                )
            except KeyError:
                pass


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        print("-" * 80)
        ds = WildDashDataset(split="train")
        print(ds)
        print(ds._build_manifest())
