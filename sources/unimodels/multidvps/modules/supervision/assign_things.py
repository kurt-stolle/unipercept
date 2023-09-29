from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tensordict import TensorDict
from torch import Tensor
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass
from uniutils.box import boxes_to_areas
from uniutils.mask import masks_to_boxes, masks_to_centers

from .gaussian import draw_gaussian, get_radius


class ThingInstances(Tensorclass):
    insts: Tensor
    indices: Tensor
    indices_mask: Tensor
    categories: Tensor
    ids: Tensor

    @property
    def valid(self) -> Tensor:
        return self[self.indices_mask]


class Things(Tensorclass):
    instances: ThingInstances
    scoremap: Tensor

    @property
    def num_instances(self) -> int:
        return int(self.instances.indices_mask.sum().item())


class ThingAssigner(nn.Module):
    def __init__(
        self,
        sem_ignore: int,
        num_cats: int,
        common_stride: int,
        gaussian_sigma: int,
        min_overlap: float,
        embeddings: dict[int, int],
        pad_to: int | None = 92,
    ):
        """
        Discovers and organized thing-class objects in the ground truth.

        Parameters
        ----------
        sem_ignore : int
            The semantic class value to ignore.
        num_cats : int
            The number of thing categories.
        common_stride : int
            The common stride of the network.
        gaussian_sigma : int
            The sigma value for the gaussian kernel.
        min_overlap : float
            The minimum overlap between a ground truth box and a predicted box.
        embeddings : dict[int, int]
            The embeddings for each thing category, i.e. translates contiguous train ID to thing-specific train ID.
        pad_to : Optional[int], optional
            The size to pad the ground truth to, by default 92. When set to None, padding is automatically determined
            based on the amount of ground truth objects.
        """
        super().__init__()

        self.sem_ignore = sem_ignore
        self.num_cats = num_cats
        self.common_stride = common_stride
        self.gaussian_sigma = gaussian_sigma
        self.min_overlap = min_overlap
        self.embeddings = embeddings
        self.pad_to = pad_to

    @staticmethod
    @lru_cache()
    def scale_limits(hw_input: tuple[int, int], levels: int, k=3, factor=1.25) -> list[tuple[int, int]]:
        power = 1 / factor
        scale = torch.linspace(0, 1**power, levels + k) ** factor
        x, y = (s * scale for s in hw_input)
        area = x * y
        area.sqrt_()
        area.ceil_()

        return torch.stack((area[:levels], area[k : levels + k]), dim=1).int().tolist()

    @override
    def forward(
        self,
        semmap: Tensor,
        insmap: Tensor,
        *,
        hw_input: torch.Size,
        hw_embedding: torch.Size,
        hw_detections: Mapping[str, torch.Size],
    ) -> list[Things]:
        # if not insmap.any():
        #     return [None for _ in hw_detections]

        assert all(a == b for a, b in zip(hw_input, semmap.shape[-2:])), f"Expected {hw_input}, got {semmap.shape}"

        semmap = self._translate(semmap, insmap)

        lblmap_div = 2**16
        lblmap = semmap * lblmap_div + insmap
        lblmap.masked_fill_(insmap == 0, 0)
        lblmap.masked_fill_(semmap == self.sem_ignore, 0)

        # Find unique labels
        lbls, lbls_to_map = lblmap.unique(return_inverse=True, sorted=True)

        # One-hot encode the inverse mapping to get a mask for each unique label
        masks = F.one_hot(lbls_to_map.reshape_as(lblmap)).float()

        # Remove the unique void label @ 0
        lbl_valid_mask = lbls > 0

        lbls = lbls[lbl_valid_mask]
        masks = masks[..., lbl_valid_mask]

        masks = rearrange(
            masks, "b h w n -> b n h w"
        )  # masks.permute(0, 3, 1, 2)  # rearrange(masks, "b h w n -> b n h w")
        has_lbl = masks.any(dim=-1).any(dim=-1)  # masks.view(masks.shape[0], masks.shape[1], -1).any(dim=-1)

        # Create an index to translate back to the original
        masks = masks[has_lbl]
        lbls = (lbls * has_lbl)[has_lbl]
        cats = torch.div(lbls, lblmap_div, rounding_mode="floor")
        iids = torch.remainder(lbls, lblmap_div)

        # Convert masks to bounding boxes
        boxes = masks_to_boxes(masks)
        areas = boxes_to_areas(boxes)
        areas.sqrt_()

        # Sample bit masks to lower scale
        h, w = masks.shape[-2:]
        h, w = int(h / self.common_stride + 0.5), int(w / self.common_stride + 0.5)

        assert masks.ndim == 3, f"Expected 4D masks, got {masks.ndim}D"
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Determine centers
        mask_centers_xy = masks_to_centers(masks, stride=1)
        mask_centers_xy.masked_fill_(~mask_centers_xy.isfinite(), 0.0)

        # Assign labels per thing
        area_limits = self.scale_limits(hw_input, len(hw_detections))
        assert all(areas <= area_limits[-1][1]), "Area of instances exceeds maximum area of feature map"

        return TensorDict.from_dict(
            {
                key: self._assign(
                    has_lbl=has_lbl,
                    boxes=boxes[areas_valid],
                    centers=mask_centers_xy[areas_valid],
                    masks=masks[areas_valid],
                    cats=cats[areas_valid],
                    iids=iids[areas_valid],
                    areas_valid=areas_valid,
                    hw=hw,
                    hw_input=hw_input,
                    hw_embedding=hw_embedding,
                )
                for areas_valid, (key, hw) in zip(
                    ((areas > area_min) & (areas <= area_max) for area_min, area_max in area_limits),
                    hw_detections.items(),
                )
            }
        )

    def _assign(
        self,
        *,
        has_lbl: Tensor,
        boxes: Tensor,
        centers: Tensor,
        masks: Tensor,
        cats: Tensor,
        iids: Tensor,
        areas_valid: Tensor,
        hw: torch.Size,
        hw_input: torch.Size,
        hw_embedding: torch.Size,
    ) -> Things:
        # Compute the scale of this feature level w.r.t. image and mask zies
        scale_input = self._compute_relative_scale(hw, hw_input)
        scale_embed = self._compute_relative_scale(hw, hw_embedding)

        # Compute the index where the center of each instance in the feature map would be
        # after scaling the mask to the feature map size
        # centers = centers[areas_valid].clone()
        # centers[..., 0] *= scale_embed[1]
        # centers[..., 1] *= scale_embed[0]

        # Sampling index, i.e. the center location of each object after flattening
        # centers_index = centers.to(torch.long, copy=True)
        centers_index = self._scale_values(centers, scale_embed).to(torch.long, copy=True)
        centers_index.floor_()
        # centers_index[:, 0].clamp_(min=0, max=hw_scoremap[1])
        # centers_index[:, 1].clamp_(min=0, max=hw_scoremap[0])
        assert torch.all(centers_index >= 0), "Negative index"

        indices = centers_index[..., 1] * hw[1] + centers_index[..., 0]
        # To translate back to (batch, num, ...) shape, we need to also transform the
        # masks valid masking
        select = has_lbl.clone()
        select[has_lbl] = areas_valid

        # Create an index for the batch dimension
        # Create thing instances
        ins, ass_idx = self._new_instances(select, has_lbl, hw_embedding=hw_embedding)
        ins_flat = ins.view(-1)
        ins_flat.indices[ass_idx] = indices
        ins_flat.insts[ass_idx] = masks
        ins_flat.categories[ass_idx] = cats
        ins_flat.ids[ass_idx] = iids

        # Draw each scoremap
        width = (boxes[..., 2] - boxes[..., 0]) * scale_input[1]
        height = (boxes[..., 3] - boxes[..., 1]) * scale_input[0]
        radii = get_radius(width, height, self.min_overlap).clamp(min=0).int()

        bat_masks = torch.ones_like(select, dtype=torch.long).cumsum(dim=0) - 1
        bat_masks = bat_masks[select]

        scoremaps = [
            self._new_scoremap(
                cats=cats[batch_mask],
                centers=centers_index[batch_mask],
                radii=radii[batch_mask],
                hw=hw,
            )
            for batch_mask in (bat_masks == i for i in range(select.shape[0]))
        ]

        ths = Things(
            instances=ins,
            scoremap=torch.stack(scoremaps),
            batch_size=ins.batch_size[:1],
        )  # type: ignore

        if not areas_valid.any():
            return ths  # empty

        return ths

    def _new_instances(
        self, select: Tensor, has_lbl: Tensor, *, hw_embedding: torch.Size
    ) -> tuple[ThingInstances, Tensor]:
        device = select.device
        batch_size = select.shape[0]

        # Amount of instances per batch
        num_per_batch = select.sum(dim=-1, keepdim=True)  # (B, 1)
        dims = self.pad_to if self.pad_to is not None else int(num_per_batch.amax().cpu().item())

        # Index for translating back to the original ordering
        off_batch = torch.ones_like(has_lbl, dtype=torch.long).cumsum_(dim=0) - 1
        off_batch *= dims
        off_count = torch.ones_like(has_lbl, dtype=torch.long)
        off_count[~select] = 0
        off_count.cumsum_(dim=1)
        off_count -= 1

        ass_idx = (off_batch + off_count)[select]  # (B * N)

        # Number of instances per batch
        ins_index = torch.ones((batch_size, dims), dtype=torch.long, device=device).cumsum_(dim=-1) - 1  # (B, pad_to)
        ins_valid = ins_index < num_per_batch  # (B, pad_to)

        # assert ass_idx.shape[0] == ins_valid.sum(), f"{ass_idx.shape[0]} != {int(ins_valid.sum())}"

        # Allocate memory for instances
        # ins = ThingInstances(
        #     indices=torch.full_like(ins_valid, -1, dtype=torch.long, device=device),
        #     indices_mask=ins_valid,
        #     insts=torch.full((*ins_valid.shape, *hw_embedding), self.sem_ignore, dtype=torch.float, device=device),
        #     categories=torch.full_like(ins_valid, self.sem_ignore, dtype=torch.long, device=device),
        #     ids=torch.full_like(ins_valid, self.sem_ignore, dtype=torch.long, device=device),
        #     batch_size=ins_valid.shape,
        # )  # type: ignore

        ins = self._allocate_empty(ins_valid.shape, hw_embedding, self.sem_ignore, ins_valid.device).clone()
        ins.indices_mask = ins_valid

        return ins, ass_idx

    @staticmethod
    @lru_cache(maxsize=16)
    def _allocate_empty(
        shape: torch.Size, hw_embedding: torch.Size, sem_ignore: int, device: torch.device
    ) -> ThingInstances:
        return ThingInstances(
            indices=torch.full(shape, -1, dtype=torch.long, device=device),
            indices_mask=torch.zeros(shape, dtype=torch.bool, device=device),
            insts=torch.full((*shape, *hw_embedding), sem_ignore, dtype=torch.float, device=device),
            categories=torch.full(shape, sem_ignore, dtype=torch.long, device=device),
            ids=torch.full(shape, sem_ignore, dtype=torch.long, device=device),
            batch_size=shape,
        )  # type: ignore

    @staticmethod
    def _compute_relative_scale(baseline: torch.Size, other: torch.Size) -> tuple[float, ...]:
        return tuple(a / b for a, b in zip(baseline, other))

    @staticmethod
    def _scale_values(tensor: Tensor, scale: tuple[float, ...]) -> Tensor:
        """Scale values along the last dimension of a tensor with a scale factor."""
        return tensor * torch.tensor(scale, dtype=tensor.dtype, device=tensor.device, requires_grad=False)

    def _new_scoremap(self, cats: Tensor, centers: Tensor, radii: Tensor, hw: torch.Size) -> Tensor:
        scoremap = torch.zeros((self.num_cats, *hw), dtype=torch.float, device=cats.device)
        for idx, radius, center in zip(
            cats,
            radii,
            centers,
        ):
            fmap = scoremap[idx]
            draw_gaussian(fmap, center, int(radius.item()), sigma_factor=self.gaussian_sigma)
            assert fmap.max() >= 0

        return scoremap

    def _translate(self, semmap: Tensor, insmap: Tensor):
        semmap_thing = torch.full_like(semmap, self.sem_ignore)
        # insmap[semmap_thing == 255]  = 0
        for e_from, e_to in self.embeddings.items():
            semmap_thing.masked_fill_(semmap == e_from, e_to)

        # assert (semmap_thing[insmap > 0] != (self.sem_ignore)).all(), "Instance map has invalid semantic labels!"

        return semmap_thing
