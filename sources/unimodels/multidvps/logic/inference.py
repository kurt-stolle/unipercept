"""Implements the inference pipeline."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from tensordict import TensorDict
from torch import Tensor
from typing_extensions import override

import unipercept as up
from unipercept.utils.function import multi_apply
from unipercept.utils.tensor import cat_nonempty, topk_score

from ..modules import DepthPrediction, Detection
from ._structures import Context, StuffInstances, ThingInstances

__all__ = ["InferencePipeline"]


class InferencePipeline(nn.Module):
    def __init__(
        self,
        *,
        inst_thres: float,
        center_top_num: int,
        center_thres: float,
        sem_thres: float,
        panoptic_overlap_thrs: float,
        panoptic_stuff_limit: int,
        panoptic_inst_thrs: float,
    ):
        super().__init__()
        self.center_thres = center_thres
        self.sem_thres = sem_thres
        self.center_top_num = center_top_num
        self.panoptic_overlap_thrs = panoptic_overlap_thrs
        self.panoptic_stuff_limit = panoptic_stuff_limit
        self.panoptic_inst_thrs = panoptic_inst_thrs
        self.inst_thres = inst_thres
        self.center_top_num = center_top_num
        self.force_predict = True

        assert 0 <= self.inst_thres <= 1, "inst_thres must be in range [0, 1]"

    @torch.jit.script_if_tracing
    def predict_things(
        self,
        ctx: Context,
    ) -> tuple[TensorDict, Tensor, Tensor]:
        # Run inference for each stage
        pool_size = [3, 3, 3, 5, 5]
        (
            nums,
            kernels,
            cats,
            scores,
        ) = multi_apply(
            self._detect_things,
            ctx.detections.values(),
            pool_size,
        )
        # Aggregate Thing classes
        num = sum(nums)
        if num == 0:
            device = ctx.embeddings.device
            return (
                kernels[0],
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )

        cats = cat_nonempty(cats, dim=1)
        scores = cat_nonempty(scores, dim=1)
        kernels = cat_nonempty(kernels, dim=1)

        assert not (cats is None or scores is None or kernels is None)

        # Sort things by their score
        sort_index = torch.argsort(scores, descending=True)
        for t in (cats, scores, kernels):
            torch.gather(t.clone(), dim=1, index=sort_index, out=t)

        return kernels, cats, scores

    def _detect_things(self, det: Detection, pool_size: int):
        """Find instances by a peak detection algorithm."""

        assert det.shape[0] == 1, "Only batch size 1 is supported during inference!"

        centers = det.thing_map.sigmoid()

        # Apply average pooling on the center locations map to smooth it.
        centers += nn.functional.avg_pool2d(
            centers,
            kernel_size=pool_size,
            stride=1,
            padding=(pool_size - 1) // 2,
        )
        centers /= 2.0

        # Select center locations via peak detection.
        fmap_max = nn.functional.max_pool2d(centers, 3, stride=1, padding=1)
        mask = (fmap_max == centers).float()
        centers *= mask

        # Ensure there are enough centers to select from, i.e. the number of
        # centers is greater than the number of centers requested.
        centers_shape = centers.shape
        top_num = min(centers_shape[-2] * centers_shape[-1], self.center_top_num // 2)

        # Keep the top-k (self.center_top_num) centers that have the largest value,
        # where the value center represents the confidence score of the center
        # location.
        sub_score, sub_index, sub_class, *_ = topk_score(centers, K=top_num, score_shape=centers_shape)

        # Apply a threshold to center locations to filter out low-confidence
        # center locations.
        mask = sub_score > self.center_thres
        scores, cats, index = (torch.masked_select(t, mask) for t in (sub_score, sub_class, sub_index))

        # Re-add the fake batch dimension of 1
        scores, cats, index = (rearrange(t, "num -> () num") for t in (scores, cats, index))

        # Amount of things detected is equal to the amount of True elements in the
        # keep list.
        num = mask.sum()

        # Kernel sampling
        index = index.to(device=centers.device, dtype=torch.long)

        def empty_kernels(k_space: Tensor) -> Tensor:
            """
            Transform the kernel space into an empty tensor of shape (batch, 0, dims).
            This implies nothing was detected.
            """
            batch, dims, _, _ = k_space.shape

            return torch.empty((batch, 0, dims), device=k_space.device, dtype=k_space.dtype)

        def sample_kernels(k_space: Tensor) -> Tensor:
            """
            Sample kernels from the kernel space, given the index of the kernels.
            Returns a tensor of shape (batch, N, dims), where N is the amount of
            detections in the index.
            """
            k_space = rearrange(k_space, "batch dims h w -> batch dims (h w)")
            i = repeat(index, "batch num -> batch dims num", dims=k_space.shape[1])
            k = torch.gather(k_space, dim=2, index=i)
            k = rearrange(k, "batch dims num -> batch num dims")

            assert k.shape[1] == num, f"{k.shape[1]} != {num}"

            return k

        kernels = det.kernel_spaces.apply(
            empty_kernels if num == 0 else sample_kernels,
            batch_size=[1, num],
        )

        return (
            num,
            kernels,
            cats,
            scores,
        )

    def predict_stuff(self, ctx: Context, **kwargs) -> T.Tuple[TensorDict, Tensor, Tensor]:
        """Infer semantic segmentation from predicted regions and kernel weights."""

        (
            kernels,
            scores,
            categories,
            num,
        ) = multi_apply(self._detect_stuff, ctx.detections.values(), **kwargs)

        # Aggregate Stuff classes
        num = sum(num)

        scores = cat_nonempty(scores, dim=1)
        categories = cat_nonempty(categories, dim=1)
        kernels = cat_nonempty(kernels, dim=1)

        return kernels, categories, scores

    def _detect_stuff(self, det: Detection, stuff_all_classes: bool = False, stuff_with_things: bool = False):
        # Region logits from the locations tensor
        regs = det.stuff_map.sigmoid()

        assert regs.shape[0] == 1, f"Expected batch size of 1, got {regs.shape[0]}"

        cats = regs.argmax(dim=1)

        # Amount of stuff classes is the amount of unique categories
        stuff_channels = det.stuff_map.shape[1]  # B (C) H W
        mask = nn.functional.one_hot(cats, num_classes=stuff_channels)
        mask = rearrange(mask, "batch h w cats -> batch cats h w").contiguous()

        # Select unique categories to find pixel amounts
        cats, cats_num = torch.unique(cats, return_counts=True)

        # Stuff score for each pixel
        # scores = (regs * mask).reshape(1, self.detector.stuff_channels, -1)
        # scores = (scores.sum(dim=-1)[:, cats] / cats_num).squeeze(0)
        scores = reduce(regs * mask, "batch cats h w -> cats", "sum")[cats] / cats_num

        # Select only masks for which a detection was made
        mask.squeeze_(0)
        mask = mask[cats]

        # Threshold to stuff pixel-level scores
        keep = scores > self.sem_thres
        num = keep.sum()

        scores = scores[keep]
        cats = cats[keep]
        mask = mask[keep]

        if not stuff_all_classes and not stuff_with_things:
            cats += 1

        # Alter dimensions for branched operations in sampling
        pixels = cats_num[keep].clamp(min=1)
        pixels.unsqueeze_(1)  # (cats 1)
        mask.unsqueeze_(1)  # (cats, 1, h, w)

        def empty_kernels(k_space: Tensor) -> Tensor:
            """
            Transform the kernel space into an empty tensor of shape (batch, 0, dims).
            This implies nothing was detected.
            """
            batch, dims, _, _ = k_space.shape

            return torch.empty((batch, 0, dims), device=k_space.device, dtype=k_space.dtype)

        def sample_kernels(k_space: Tensor) -> Tensor:
            assert k_space.shape[0] == 1, f"Expected batch size of 1, got {k_space.shape[0]}"

            k = mask * k_space
            k = reduce(k, "num dims h w -> num dims", "sum") / pixels
            k.unsqueeze_(0)

            return k  # (1, num, dims)

        kernels = det.kernel_spaces.apply(
            empty_kernels if num == 0 else sample_kernels,
            batch_size=[1, num],
        )

        # Add back fake batch dimension for consistency between interfaces
        scores.unsqueeze_(0)
        cats.unsqueeze_(0)

        return (
            kernels,
            scores,
            cats,
            num,
        )

    def merge_predictions(
        self,
        ctx: Context,
        things: ThingInstances,
        stuff: StuffInstances,
        *,
        stuff_all_classes: bool,
        stuff_with_things: bool,
        id_map_thing: T.Mapping[int, int],
        id_map_stuff: T.Mapping[int, int],
    ) -> tuple[up.data.tensors.PanopticMap, up.data.tensors.DepthMap, ThingInstances]:
        """Combine unbatched things and stuff."""

        assert ctx.device is not None, f"Device should be configured in context"
        # First and only element of the 'batch' is the amount of thing/stuff detections
        assert len(things.batch_size) == 1
        assert len(stuff.batch_size) == 1

        stuff_num = len(id_map_stuff)
        if not (stuff_with_things or stuff_all_classes):
            stuff_num += 1

        sem_logits = torch.zeros((stuff_num, *ctx.input_size), device=stuff.device)  # type: ignore
        if stuff.num_instances > 0:  # if detections have been made
            sem_logits[stuff.categories] += self.upscale_to_input_size(ctx, stuff.logits).squeeze_(0)
        sem_seg = sem_logits.argmax(dim=0)

        # Allocate memory for flat outputs
        out_ins = torch.zeros_like(sem_seg, dtype=torch.int32)
        out_sem = torch.full_like(
            sem_seg,
            up.data.tensors.PanopticMap.IGNORE,
            dtype=torch.int32,
            device=ctx.device,
        )
        out_depth = torch.zeros_like(sem_seg, dtype=torch.float32)

        # Panoptic results for output without void
        pan_logits = []
        pan_ins = []
        pan_sem = []
        pan_depths = []

        # Filter instances below threshold value for score
        if things is not None and things.num_instances > 0:
            things_thrs = things.scores > self.panoptic_inst_thrs
            things = T.cast(ThingInstances, things.get_sub_tensordict(things_thrs))

        # Filter instances by thresholding score
        if things is not None and things.num_instances > 0:
            # Upsample instance masks and logits
            thing_logits = self.upscale_to_input_size(ctx, things.logits)[0]
            thing_masks = thing_logits > self.inst_thres

            # Add instances one-by-one, checking for overlaps with existing
            for idx in range(things.num_instances):
                out_free = out_sem == up.data.tensors.PanopticMap.IGNORE

                mask = thing_masks[idx]
                mask_area = mask.sum().item()
                intersect = mask & (~out_free)
                intersect_area = intersect.sum().item()

                # Determine whether the mask interest is above the set threshold
                if mask_area == 0 or intersect_area * 1.0 / mask_area > self.panoptic_overlap_thrs:
                    continue

                # Refine the mask to not include intersecting segments
                # This is not updated back to the thing object itself
                if intersect_area > 0:
                    mask = mask & (out_free)

                # Instance ID is set to the index plus one, since zero indices
                # a stuff or crowd label
                ins_cat_train = things.categories[idx].item()  # type: ignore
                assert isinstance(ins_cat_train, int), type(ins_cat_train)
                ins_cat = id_map_thing[ins_cat_train]
                ins_idx = idx + 1

                # Translate train ID to contiguous ID for outputs
                out_sem[mask] = ins_cat
                out_ins[mask] = ins_idx

                # Gather depth mask for this instance
                if things.depths is not None:
                    ins_depth = things.depths.maps[idx]
                    out_depth[mask] = ins_depth[mask]
                else:
                    ins_depth = torch.zeros_like(mask, dtype=torch.float32)

                # Append
                if self.force_predict:
                    pan_logits.append(thing_logits[idx])
                    pan_ins.append(ins_idx)
                    pan_sem.append(ins_cat)
                    pan_depths.append(ins_depth)

        # Add semantic predictions one-by-one
        sem_cats: list[int] = torch.unique(sem_seg).tolist()
        sem_cats.reverse()

        for cat_st_train in sem_cats:
            # cat_st_train = cat_st_train.item()
            cat_st = id_map_stuff[cat_st_train]

            # Check skipping conditions based on model config
            if stuff_with_things and cat_st_train == 0:
                continue  # 0 is a special 'thing' class
            # if stuff_all_classes and (cat_st in id_map_thing.values()):
            #     continue  # Skip semantic classes that are also things

            # Select only pixels that belong to the current class and are not
            # already present in the output panpotic segmentation
            out_free = (out_sem == up.data.tensors.PanopticMap.IGNORE) | (out_sem == cat_st_train)

            mask = (sem_seg == cat_st_train) & out_free
            mask_area = mask.sum().item()

            # Determine area threshold based on configuration
            if cat_st not in id_map_thing.values():
                # Mask is a STUFF class
                if mask_area < self.panoptic_stuff_limit:
                    continue  # Enforce minimal stuff area
            # elif cat_st not in self.meta.cats_tracked:
            #     # Mask is a non-tracked THING class
            #     intersect_area = mask.sum().item()

            #     # Determine whether the mask interest is above the set threshold
            #     if mask_area == 0 or intersect_area * 1.0 / mask_area > self.panoptic_overlap_thrs:
            #         continue
            else:
                # Mask is a THING class (and should be ignored here)
                continue

            # Gather stuff depth mask as the mean value over all detected channels
            stuff_cats_idx = stuff.categories == cat_st_train
            if stuff.depths is None:
                sem_depth = torch.zeros_like(mask, dtype=torch.float32)
            else:
                sem_depth = stuff.depths.maps[stuff_cats_idx]
            sem_depth = reduce(sem_depth, "n h w -> h w", "mean")

            # Insert current segment to masks
            out_ins[mask] = 0
            out_sem[mask] = cat_st
            out_depth[mask] = sem_depth[mask]

            # Append lists
            if self.force_predict:
                # if True:
                pan_logits.append(sem_logits[cat_st_train])
                pan_sem.append(cat_st)
                pan_ins.append(0)
                pan_depths.append(sem_depth)

        # Return if there are no detections
        if len(pan_logits) == 0:
            return (
                up.data.tensors.PanopticMap.from_parts(out_sem, out_ins),
                out_depth.as_subclass(up.data.tensors.DepthMap),
                things,
            )

        # Gather final depth mapping using overall argmax
        pan_logits = torch.stack(pan_logits).to(ctx.device)  # type: ignore
        pan_select = pan_logits.argmax(dim=0)
        pan_depths = torch.stack(pan_depths).to(ctx.device)  # type: ignore

        # Old version: select depth at specific mask
        out_depth = torch.gather(pan_depths, dim=0, index=pan_select.unsqueeze(0)).squeeze(0)

        # New version: select depth by using a weighted mean of the depth logits
        # out_depth = torch.sum(pan_depths * pan_logits.softmax(dim=-1), dim=0)

        if self.force_predict:
            # In force predict mode, we always outptu a prediction, even if the
            # label is void
            pan_sem = torch.as_tensor(pan_sem, dtype=torch.int32, device=ctx.device)
            pan_ins = torch.as_tensor(pan_ins, dtype=torch.int32, device=ctx.device)

            # Overwrite semantic and instances using panoptic logit-based argmax
            out_sem = pan_sem[pan_select]
            out_ins = pan_ins[pan_select]
        # else:
        #     out_depth = torch.where(
        #         out_depth == 0,
        #         torch.gather(pan_depths, dim=0, index=pan_select.unsqueeze(0)).squeeze(0),
        #         out_depth,
        #     )

        # Select instances that were not dropped during the creation of the
        # panoptic map
        ins_uniq = torch.unique(out_ins)
        ins_keep = (ins_uniq[ins_uniq > 0] - 1).tolist()

        if things is None or len(ins_keep) == 0:
            return (
                up.data.tensors.PanopticMap.from_parts(out_sem, out_ins),
                out_depth.as_subclass(up.data.tensors.DepthMap),
                things,
            )
        else:
            # Infer the instance ID via a tracking algorithm
            things = things.get_sub_tensordict([ins_keep])

            return (
                up.data.tensors.PanopticMap.from_parts(
                    out_sem,
                    out_ins,
                ),
                out_depth.as_subclass(up.data.tensors.DepthMap),
                things,
            )

    def upscale_to_input_size(
        self,
        ctx: Context,
        mask: Tensor,
    ) -> Tensor:
        """Upscale a mask to the input size of the model."""

        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        mask_up = nn.functional.interpolate(
            mask,
            ctx.captures.images.shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        if mask_up.ndim > mask.ndim:
            mask_up.squeeze_(0)
            assert (
                mask_up.ndim == mask.ndim
            ), f"Mask has wrong number of dimensions: {list(mask_up.shape)} vs {list(mask.shape)}"

        return mask_up
