"""Implements the inference pipeline."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from tensordict import TensorDictBase
from torch import Tensor
from typing_extensions import override
from unipercept.data import points as data_points
from unipercept.modeling import layers as modules
from uniutils.function import multi_apply
from uniutils.tensor import cat_nonempty, topk_score

from ..keys import KEY_DEPTH
from ..modules import DepthPrediction, Detection
from ._structures import (
    Context,
    PanopticMap,
    SampleInfo,
    StuffInstances,
    ThingInstances,
)

if T.TYPE_CHECKING:
    from unipercept.data.points import InputData


__all__ = ["InferencePipeline", "InferenceResult"]

InferenceResult: T.TypeAlias = T.Dict[str, T.Any]


class InferencePipeline(nn.Module):
    def __init__(self, *, inst_thres: float, output_keys: T.Optional[T.Sequence[str]] = None, **kwargs):
        super().__init__(**kwargs)

        self.output_keys = set(output_keys or ["panoptic_seg", "depth"])
        assert len(self.output_keys) > 0, "Must request at least one output key."

        self.inst_thres = inst_thres
        assert 0 <= self.inst_thres <= 1, "inst_thres must be in range [0, 1]"

    def __inference_batch_item(
        self,
        ctx: Context,
        ifo: SampleInfo,
    ) -> dict[str, T.Any]:
        # Upscale depth feature
        if KEY_DEPTH in ctx.embeddings.keys():
            feat_depth = ctx.embeddings.get(KEY_DEPTH)
            feat_depth = self.__upscale_mask(ctx, ifo, feat_depth)

            ctx.embeddings = ctx.embeddings.set(KEY_DEPTH, feat_depth)

        # Infer individual object categoires (things)
        things = self.__infer_thing(ctx)

        # Infer background/static categories (stuff)
        stuff = self.__infer_stuff(ctx)
        # Combine stuff and things into a single panoptic segmentation
        combined = self.__combine_merge(
            ctx,
            ifo,
            things,
            stuff,
        )

        # Compile result based on requested outputs
        output = {}
        for req in self.output_keys:
            if req in output:
                raise ValueError(f"Duplicate output key: {req}")

            if req == "panoptic_labels":
                res = (combined.semantic, combined.instance)
            elif req == "depth":
                res = data_points.DepthMap(combined.depth)
            elif req == "instance":
                res = data_points.Mask(combined.instance)
            elif req == "semantic":
                res = data_points.Mask(combined.semantic)
            elif req == "panoptic":
                semantic = combined.semantic
                semantic[semantic == self.ignored_label] = -1
                res = data_points.PanopticMap.from_parts(semantic, combined.instance)
            # elif req == "panoptic_seg":
            #     # Recent version of D2 support a label format that is equivalent
            #     # to the canonical format, but with the ignored label being -1
            #     panoptic_seg = combined.semantic * self.label_divisor
            #     panoptic_seg += combined.instance
            #     panoptic_seg[combined.semantic == self.ignored_label] = -1

            #     res = (panoptic_seg, None)
            else:
                raise ValueError(f"Unknown output key: {req}")

            output[req] = res

        return output

    def __infer_thing(
        self,
        ctx: Context,
    ) -> ThingInstances | None:
        # Run inference for each stage
        pool_size = [3, 3, 3, 5, 5]
        (
            nums,
            kernels,
            cats,
            scores,
        ) = multi_apply(
            self.__infer_thing_level,
            ctx.detections.values(),
            pool_size,
        )
        # Aggregate Thing classes
        num = sum(nums)
        if num == 0:
            return None

        cats = cat_nonempty(cats, dim=1)
        scores = cat_nonempty(scores, dim=1)
        kernels = cat_nonempty(kernels, dim=1)

        assert not (cats is None or scores is None or kernels is None)

        # Sort things by their score
        sort_index = torch.argsort(scores, descending=True)
        for t in (cats, scores, kernels):
            torch.gather(t.clone(), dim=1, index=sort_index, out=t)

        # Generate things
        kernels, cats, scores = self.fusion_thing(kernels, cats, scores)
        thing_masks = self.maskifier_thing(
            features=ctx.embeddings,
            kernels=kernels,
        ).sigmoid_()

        # Generate depth for detected instances
        if self.depth_mapper is not None:
            dmap, means = self.depth_mapper(features=ctx.embeddings, kernels=kernels)
            depth = DepthPrediction(maps=dmap, means=means, batch_size=kernels.batch_size, device=None).view(-1)
        else:
            depth = None

        masks = thing_masks > self.inst_thres

        score_factor = (thing_masks * masks.float()).sum((-2, -1))
        score_factor /= masks.sum((-2, -1)).float().clamp(1e-8)
        scores *= score_factor

        things = ThingInstances(
            kernels=kernels,
            masks=thing_masks,
            scores=scores,
            categories=cats,
            iids=torch.zeros_like(cats),
            depths=depth,
            batch_size=kernels.batch_size,
        )

        # Rebalance the score from the mask generator with the mask size

        # Sorting index by score
        sort_index = torch.argsort(things.scores, descending=True)
        # things.apply(lambda x: torch.index_select(x, 0, sort_index), inplace=True)
        things = things[sort_index]

        # Keep only scores above a threshold
        keep_mask = things.masks.scores >= 0.05
        if not keep_mask.any():
            return None

        things = things.masked_select(keep_mask)

        # Sort again and keep only the top instances
        sort_index = torch.argsort(things.masks.scores, descending=True)
        sort_index = sort_index[: self.center_top_num]
        things = things[sort_index]

        return things.contiguous()

    def __infer_thing_level(self, det: Detection, pool_size: int):
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
        index = index.to(device=self.device, dtype=torch.long)

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

    def __infer_stuff(self, ctx: Context) -> StuffInstances:
        """Infer semantic segmentation from predicted regions and kernel weights."""

        (
            kernels,
            scores,
            categories,
            num,
        ) = multi_apply(
            self.__infer_stuff_level,
            ctx.detections.values(),
        )

        # Aggregate Stuff classes
        num = sum(num)

        scores = cat_nonempty(scores, dim=1)
        categories = cat_nonempty(categories, dim=1)
        kernels = cat_nonempty(kernels, dim=1)

        # Fusion
        kernels, categories, scores = self.fusion_stuff(kernels, categories, scores)

        # Generate semantic predictions
        stuff_masks = self.maskifier_stuff(
            features=ctx.embeddings,
            kernels=kernels,
            categories=categories,
            scores=scores,
        ).sigmoid_()
        # Generate depth for detected instances
        if self.depth_mapper is not None:
            dmap, means = self.depth_mapper(features=ctx.embeddings, kernels=kernels)
            assert dmap is not None
            depth = DepthPrediction(maps=dmap, means=means, batch_size=kernels.batch_size).view(-1)
        else:
            depth = None

        # Convert to results object, which requires removal of the fake batch dimension
        stuff = StuffInstances(  # type: ignore
            masks=stuff_masks.view(-1),
            depths=depth,
            batch_size=gen.batch_size,
            device=gen.device,
        )

        return stuff.contiguous()

    def __infer_stuff_level(self, det: modules.Detection):
        # Region logits from the locations tensor
        regs = det.stuff_map.sigmoid()

        assert regs.shape[0] == 1, f"Expected batch size of 1, got {regs.shape[0]}"

        cats = regs.argmax(dim=1)

        # Amount of stuff classes is the amount of unique categories in the
        mask = nn.functional.one_hot(cats, num_classes=self.detector.localizer.stuff_channels)
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

        if not self.stuff_all_classes and not self.stuff_with_things:
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

    def __combine_merge(
        self,
        ctx: Context,
        ifo: SampleInfo,
        things: T.Optional[ThingInstances],
        stuff: StuffInstances,
    ) -> PanopticMap:
        """Combine unbatched things and stuff."""

        # First and only element of the 'batch' is the amount of thing/stuff detections
        assert things is None or len(things.batch_size) == 1
        assert len(stuff.batch_size) == 1

        if self.stuff_with_things or self.stuff_all_classes:
            stuff_num = self.detector.localizer.stuff_channels
        else:
            stuff_num = self.detector.localizer.stuff_channels + 1

        sem_logits = torch.zeros((stuff_num, *ifo.size), device=self.device)
        if stuff.num_instances > 0:  # if detections have been made
            sem_logits[stuff.masks.categories] += self.__upscale_mask(ctx, ifo, stuff.masks.logits.unsqueeze(0))[0]
        sem_seg = sem_logits.argmax(dim=0)

        # Allocate memory for flat outputs
        out_ins = torch.zeros_like(sem_seg, dtype=torch.int32)
        out_sem = torch.full_like(
            sem_seg,
            self.ignored_label,
            dtype=torch.int32,
            device=self.device,
        )
        out_depth = torch.zeros_like(sem_seg, dtype=torch.float32)

        # Panoptic results for output without void
        pan_logits = []
        pan_ins = []
        pan_sem = []
        pan_depths = []

        # Filter instances below threshold value for score
        if things is not None and things.num_instances > 0:
            things = T.cast(ThingInstances, things.get_sub_tensordict(things.masks.scores >= self.panoptic_inst_thrs))

        # Filter instances by thresholding score
        if things is not None and things.num_instances > 0:
            # Upsample instance masks and logits
            thing_logits = self.__upscale_mask(ctx, ifo, things.masks.logits.unsqueeze(0))[0]
            thing_masks = thing_logits > self.inst_thres

            # Add instances one-by-one, checking for overlaps with existing
            for idx in range(things.num_instances):
                out_free = out_sem == self.ignored_label

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
                ins_cat_train = things.masks.categories[idx].item()  # type: ignore
                assert isinstance(ins_cat_train, int), type(ins_cat_train)
                ins_cat = self.id_map_thing[ins_cat_train]
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
                # if self.force_predict:
                if True:
                    pan_logits.append(thing_logits[idx])
                    pan_ins.append(ins_idx)
                    pan_sem.append(ins_cat)
                    pan_depths.append(ins_depth)

        # Add semantic predictions one-by-one
        sem_cats: list[int] = torch.unique(sem_seg).tolist()
        sem_cats.reverse()

        for cat_st_train in sem_cats:
            # cat_st_train = cat_st_train.item()
            cat_st = self.id_map_stuff[cat_st_train]

            # Check skipping conditions based on model config
            if self.stuff_with_things and cat_st_train == 0:
                continue  # 0 is a special 'thing' class
            if self.stuff_all_classes and (cat_st in self.id_map_thing.values()):
                continue  # Skip semantic classes that are also things

            # Select only pixels that belong to the current class and are not
            # already present in the output panpotic segmentation
            out_free = (out_sem == self.ignored_label) | (out_sem == cat_st_train)

            mask = (sem_seg == cat_st_train) & out_free
            mask_area = mask.sum().item()

            # Determine area threshold based on configuration
            if cat_st not in self.id_map_thing.values():
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
            stuff_cats_idx = stuff.masks.categories == cat_st_train
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
            # if self.force_predict:
            if True:
                pan_logits.append(sem_logits[cat_st_train])
                pan_sem.append(cat_st)
                pan_ins.append(0)
                pan_depths.append(sem_depth)

        # Return if there are no detections
        if len(pan_logits) == 0:
            return PanopticMap(
                semantic=out_sem,
                instance=out_ins,
                depth=out_depth,
                batch_size=out_sem.shape,
                device=self.device,
            )

        # Gather final depth mapping using overall argmax
        pan_logits = torch.stack(pan_logits).to(self.device)
        pan_select = pan_logits.argmax(dim=0)
        pan_depths = torch.stack(pan_depths).to(self.device)
        out_depth = torch.gather(pan_depths, dim=0, index=pan_select.unsqueeze(0)).squeeze(0)

        if self.force_predict:
            # In force predict mode, we always outptu a prediction, even if the
            # label is void
            pan_sem = torch.as_tensor(pan_sem, dtype=torch.int32, device=self.device)
            pan_ins = torch.as_tensor(pan_ins, dtype=torch.int32, device=self.device)

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
            return PanopticMap(
                semantic=out_sem, instance=out_ins, depth=out_depth, batch_size=out_sem.shape, device=self.device
            )
        else:
            # Infer the instance ID via a tracking algorithm
            things = things.get_sub_tensordict([ins_keep])
            things.iids = self.__assign_tracks(ctx, ifo, things).type_as(things.iids)

            out_ins_tracked = torch.zeros_like(out_ins)
            for idx, id_ in enumerate(things.iids):
                idx = idx + 1
                out_ins_tracked[out_ins == idx] = id_.type_as(out_ins_tracked)

            return PanopticMap(
                semantic=out_sem,
                instance=out_ins_tracked,
                depth=out_depth,
                batch_size=out_sem.shape,
                device=self.device,
            )

    def __assign_tracks(
        self,
        ctx: Context,
        ifo: SampleInfo,
        things: ThingInstances,
    ) -> Tensor:
        # Mock tracker that assigns an image-level unique ID to each instance.
        if self.tracker is None:
            num = things.num_instances
            return (
                torch.tensor(
                    list(range(num)),
                    device=self.device,
                )
                + 1
            )

        if ifo.id.sequence is None or ifo.id.frame is None:
            raise ValueError(f"Cannot track instances without sequence/frame ID. Got {ifo.id}.")

        # Actual tracking
        return self.tracker(ctx, things, key=ifo.id.sequence, frame=ifo.id.frame)

    def __upscale_mask(
        self,
        ctx: Context,
        ifo: SampleInfo,
        mask: Tensor,
    ) -> Tensor:
        return upscale_mask(ctx, ifo, mask, self.common_stride)

    @override
    def forward(self, input: InputData) -> TensorDictBase:
        raise NotImplementedError("Pipeline is not callable directly.")


def upscale_mask(
    ctx: Context,
    ifo: SampleInfo,
    mask: Tensor,
    scale: int,
) -> Tensor:
    img_shape = ctx.size
    ori_shape = ifo.size

    # Upscale segmentation mask using interpolation, using a scale
    # factor equal to the common stride of the downsampling layers in
    # the backbone network.
    mask_up = nn.functional.interpolate(
        mask,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[..., : img_shape[0], : img_shape[1]]

    # Interpolate the upscaled segmentation mask to match the original
    # shape of the input.
    mask_up = nn.functional.interpolate(
        mask_up,
        size=ori_shape,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    if mask_up.ndim > mask.ndim:
        mask_up.squeeze_(0)
        assert mask_up.ndim == mask.ndim, f"Mask has wrong number of dimensions: {mask_up.ndim} vs {mask.ndim}"

    return mask_up
