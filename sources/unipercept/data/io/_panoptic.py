from __future__ import annotations

from typing import Any, Final, Literal, Optional, TypeVar, cast

import torch
from unicore import file_io
from unipercept.data.points import PanopticMap
from unipercept.data.types import LabelsFormat, Metadata

from .utils import get_kwd, read_pixels

__all__ = ["read_panoptic_map"]

BYTE_OFFSET: Final[Literal[256]] = 256


@torch.inference_mode()
@file_io.with_local_path(force=True)
def read_panoptic_map(path: str, info: Metadata | None, /, **meta_kwds) -> PanopticMap:
    """Read a panoptic map from a file."""
    panoptic_format = get_kwd(meta_kwds, "format", LabelsFormat)

    match panoptic_format:
        case LabelsFormat.CITYSCAPES:
            assert info is not None
            divisor = info["label_divisor"]
            ignore_label = info["ignore_label"]
            translations = info["translations_dataset"]

            img = read_pixels(path, color=True)

            map_ = img[:, :, 0] + BYTE_OFFSET * img[:, :, 1] + BYTE_OFFSET * BYTE_OFFSET * img[:, :, 2]
            map_ = torch.where(map_ > 0, map_, ignore_label)
            map_ = torch.where(map_ < divisor, map_ * divisor, map_ + 1)

            labels = PanopticMap.from_combined(map_, divisor)
            labels.translate_semantic_(
                translation=translations,
                ignore_label=ignore_label,
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

            labels = PanopticMap.from_parts(sem, ids)

        case LabelsFormat.KITTI:
            img = read_pixels(path, color=True)
            sem = img[:, :, 0]  # R-channel
            ids = torch.add(
                img[:, :, 1] * BYTE_OFFSET,  # G channel
                img[:, :, 2],  # B channel
            )

            labels = PanopticMap.from_parts(sem, ids)
        case LabelsFormat.VISTAS:
            assert info is not None
            divisor = info["label_divisor"]
            ignore_label = info["ignore_label"]
            translations = info["stuff_translations"]

            img = read_pixels(path, color=True)
            assert img.dtype == torch.int32, img.dtype

            labels = PanopticMap.from_combined(img, divisor)
            labels.translate_semantic_(
                translation=translations,
                ignore_label=ignore_label,
            )
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

            labels = PanopticMap.from_parts(sem, ids)
            labels.translate_semantic_(
                translation=translations,
                ignore_label=ignore_label,
            )
        case _:
            raise NotImplementedError("label_type = '{label_type}'")

    assert labels is not None, f"No labels were read from '{path}' (format: {panoptic_format})"

    if len(meta_kwds) > 0:
        raise TypeError(f"Unexpected keyword arguments: {tuple(meta_kwds.keys())}")

    return labels


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
