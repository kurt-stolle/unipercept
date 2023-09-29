from functools import partial
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from unipercept.data.points import PanopticMap

from .assign_stuff import StuffAssigner
from .assign_things import ThingAssigner
from .truths import Truths


class TruthGenerator(nn.Module):
    def __init__(
        self,
        common_stride: int,
        thing_embeddings: dict[int, int],
        stuff_embeddings: dict[int, int],
        stuff_with_things: bool,
        stuff_all_classes: bool,
        min_overlap: float,
        gaussian_sigma: int,
        label_divisor: int,
        ignore_val: int = -1,
    ):
        super().__init__()

        # self.ignore_val = ignore_val
        self.common_stride = common_stride
        self.thing_embeddings = thing_embeddings
        self.stuff_embeddings = stuff_embeddings
        self.stuff_with_things = stuff_with_things
        self.stuff_all_classes = stuff_all_classes
        self.min_overlap = min_overlap
        self.gaussian_sigma = gaussian_sigma
        # self.label_divisor = label_divisor
        self.eps = 1e-6

        ignore_val = PanopticMap.IGNORE
        label_divisor = PanopticMap.DIVISOR

        self.stuff_assigner = StuffAssigner(
            ignore_val=ignore_val,
            num_cats=self.num_stuff,
            with_things=self.stuff_with_things,
            all_classes=self.stuff_all_classes,
            embeddings=self.stuff_embeddings,
        )
        self.thing_assigner = ThingAssigner(
            sem_ignore=ignore_val,
            common_stride=self.common_stride,
            gaussian_sigma=self.gaussian_sigma,
            min_overlap=self.min_overlap,
            num_cats=self.num_thing,
            embeddings=self.thing_embeddings,
            pad_to=160,
        )

        self.downsample = partial(downsample, stride=self.common_stride)

    @property
    def num_stuff(self):
        return len(self.stuff_embeddings)

    @property
    def num_thing(self):
        return len(self.thing_embeddings)

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.float16, enabled=True)
    def generate_panoptic(
        self,
        lblmap: Tensor,
        *,
        hw_image: torch.Size,
        hw_embedding: torch.Size,
        hw_detections: Mapping[str, torch.Size],
    ) -> Truths:
        """
        Generate ground truth of multi-stages according to the input.
        """

        lblmap = lblmap.as_subclass(PanopticMap)

        semmap = lblmap.get_semantic_map()
        stuff = self.stuff_assigner(
            semmap,
            hw_detections=hw_detections,
            hw_embedding=hw_embedding,
        )

        insmap = lblmap.get_instance_map()
        thing = self.thing_assigner(
            semmap,
            insmap,
            hw_detections=hw_detections,
            hw_embedding=hw_embedding,
            hw_input=hw_image,
        )

        return Truths(
            thing=thing,
            stuff=stuff,
            semmap=self.downsample(semmap),
            insmap=self.downsample(insmap),
            # labels=self.downsample(insmap),
            batch_size=lblmap.shape[:-2],
        )

    @torch.no_grad()
    def generate_depth(
        self,
        depths: Tensor,
        hw_embedding: torch.Size,
    ) -> Tensor:
        truths = []
        for depth in depths.unbind():
            scale = 1.0 / self.common_stride
            h, w = depth.shape[-2:]
            new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)

            depth_raw = F.interpolate(
                depth.float().unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            _, h, w = depth_raw.shape

            truth = torch.zeros(1, *hw_embedding, device=depths.device)
            truth[:, :h, :w] = depth_raw
            truths.append(truth)

        return torch.stack(truths, dim=0)


def downsample(img: Tensor, stride: int) -> Tensor:
    """
    Downsample a mask via the nearest-exact neighbours algorithm
    """
    h, w = img.shape[-2:]
    new_h, new_w = int(h / stride + 0.5), int(w / stride + 0.5)

    ndim = img.ndim
    if ndim == 3:
        img = img.unsqueeze(1)
    elif ndim != 4:
        raise ValueError(f"Image must be 3 or 4 dimensional, got {ndim}!")

    img_ds = F.interpolate(img.float(), size=(new_h, new_w), mode="nearest").type_as(img)

    if ndim == 3:
        img_ds = img_ds.squeeze(1)

    return img_ds
