"""Implements the main MultiDVPS model class."""

from __future__ import annotations

import math
import typing as T

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override
from unimodels.multidvps import logic
from uniutils.function import multi_apply

if T.TYPE_CHECKING:
    from tensordict import TensorDictBase
    from unimodels.multidvps.modules import (
        DepthHead,
        Detector,
        FeatureEncoder,
        MaskHead,
        StuffFusion,
        ThingFusion,
    )
    from unimodels.multidvps.modules.supervision import Truths
    from unipercept.data.points import InputData
    from unipercept.data.types import HW, BatchType
    from unipercept.modeling import layers
    from unipercept.modeling.backbones import Backbone
    from unipercept.modeling.layers.tracking import StatefulTracker


__all__ = ["MultiDVPS"]


class MultiDVPS(nn.Module):
    """Depth-Aware Video Panoptic Segmentation model using dynamic convolutions."""

    id_map_thing: T.Final[T.Mapping[int, int]]
    id_map_stuff: T.Final[T.Mapping[int, int]]
    stuff_with_things: T.Final[bool]
    stuff_all_classes: T.Final[bool]
    depth_fixed: T.Final[dict[int, float]]

    def __init__(
        self,
        *,
        common_stride: int,
        center_top_num: int,
        weighted_num: int,
        center_thres: float,
        sem_thres: float,
        panoptic_overlap_thrs: float,
        panoptic_stuff_limit: float,
        panoptic_inst_thrs: float,
        backbone: Backbone,
        detector: Detector,
        fusion_thing: ThingFusion,
        fusion_stuff: StuffFusion,
        feature_encoder: FeatureEncoder,
        maskifier_thing: MaskHead,
        maskifier_stuff: MaskHead,
        training_pipeline: logic.training.TrainingPipeline,
        inference_pipeline: logic.inference.InferencePipeline,
        id_map_thing: T.Mapping[int, int],
        id_map_stuff: T.Mapping[int, int],
        stuff_with_things: bool,
        stuff_all_classes: bool,
        depth_mapper: DepthHead,
        force_predict=False,
        depth_fixed: T.Optional[dict[int, float]] = None,
        tracker: T.Optional[StatefulTracker] = None,
    ):
        super().__init__()

        # Properties
        self.weighted_num = weighted_num
        self.center_top_num = center_top_num
        self.center_thres = center_thres
        self.sem_thres = sem_thres
        self.panoptic_overlap_thrs = panoptic_overlap_thrs
        self.panoptic_stuff_limit = panoptic_stuff_limit
        self.panoptic_inst_thrs = panoptic_inst_thrs
        self.common_stride = common_stride
        self.force_predict = force_predict

        # Metadata readable from the dataset or passed as literal
        self.id_map_thing = {int(k): int(v) for k, v in id_map_thing.items()}
        self.id_map_stuff = {int(k): int(v) for k, v in id_map_stuff.items()}
        self.stuff_with_things = stuff_with_things
        self.stuff_all_classes = stuff_all_classes

        # Categories for which the depth always has a fixed value, e.g. the sky.
        self.depth_fixed = {} if depth_fixed is None else depth_fixed

        # Submodules
        self.backbone = backbone
        self.training_pipeline = training_pipeline
        self.inference_pipeline = inference_pipeline
        self.detector = detector
        self.fusion_thing = fusion_thing
        self.fusion_stuff = fusion_stuff
        self.feature_encoder = feature_encoder
        self.maskifier_thing = maskifier_thing
        self.maskifier_stuff = maskifier_stuff
        self.depth_mapper = depth_mapper
        self.tracker = tracker

    @override
    def forward(self, inputs: InputData) -> dict[str, T.Any] | list[dict[str, T.Any]]:
        """
        Implements the forward logic for the full model.

        Backbone feature extraction, detection/kernel generation, and high-resolution feature embedding generation
        are shared across training and inference.

        Further processing is split into training and inference branches, see ``_forward_training`` and
        ``_forward_inference``.
        """
        features = self.backbone(inputs.captures.images)

        highres = self.feature_encoder(features)
        detections = self.detector(features)

        # Create combined objects context
        ctx = logic.Context(
            detections=detections,
            embeddings=highres,
            batch_size=inputs.batch_size,
        )

        # ifs = self._read_info(batched_inputs)

        # Return losses (training) or predictions (inference)
        if self.training:
            return self._forward_training(inputs, ctx)
        else:
            return self._forward_inference(inputs, ctx)

    # ---------------- #
    # Training methods #
    # ---------------- #

    def _forward_training(self, inputs: InputData, ctx: logic.Context) -> logic.training.TrainingResult:
        """Implements the forward logic for training."""

        # Results dictionary
        result = {}

        gt_seg = self.training_pipeline.true_segmentation(inputs, ctx)
        multidets, true_thing, true_stuff = tuple(
            zip(*[(ctx.detections[key], gt_seg.thing[key], gt_seg.stuff[key]) for key in self.detector.in_features])
        )

        # ============= #
        # Kernel fusion #
        # ============= #

        # Thing generation
        thing_kernels_multi, thing_nums, thing_weights = multi_apply(
            logic.training.detect_things,
            multidets,
            true_thing,
        )
        thing_kernels: TensorDict = torch.cat(thing_kernels_multi, dim=1)  # type: ignore
        thing_kernels, _, _ = self.fusion_thing(thing_kernels, None, None)
        thing_num = sum(thing_nums)
        thing_weights = torch.cat(thing_weights, dim=1)

        # Stuff generation
        stuff_kernels_multi, stuff_nums = multi_apply(
            logic.training.detect_stuff,
            multidets,
            true_stuff,
        )
        # stuff_num = sum(stuff_nums)
        stuff_kernels: TensorDict = torch.cat(stuff_kernels_multi, dim=1)  # type: ignore
        stuff_kernels, _, _ = self.fusion_stuff(stuff_kernels, None, None)

        # ================== #
        # Thing segmentation #
        # ================== #
        (
            thing_gt,
            thing_gt_idx,
            thing_gt_num,
        ) = gt_seg.mask_instances({k: n for k, n in zip(list(self.detector.in_features), thing_nums)})

        # Compute segmentation loss
        thing_logits = self.maskifier_thing(
            features=ctx.embeddings,
            kernels=thing_kernels,
        )

        if thing_gt_num > 0:
            loss_thing = self.loss_segment_things(
                x=thing_logits,
                y=thing_gt,
                y_num=thing_gt_num,
                y_mask=None,
                index_mask=thing_gt_idx.reshape(-1),
                instance_num=thing_num,
                weights=thing_weights,  # type: ignore
                weight_num=self.weighted_num,
            )
            loss_thing = loss_thing / max(thing_gt_num, 1)
        else:
            loss_thing = thing_logits.sum() * 0.0

        result["loss_seg/things"] = loss_thing

        # ================== #
        # Stuff segmentation #
        # ================== #

        stuff_logits = self.maskifier_stuff(
            features=ctx.embeddings,
            kernels=stuff_kernels,
        )

        stuff_gt, stuff_gt_idx, stuff_gt_num = gt_seg.mask_semantic(
            {k: n for k, n in zip(list(self.detector.in_features), stuff_nums)}
        )
        if stuff_gt_num > 0:
            loss_stuff = self.training_pipeline.loss_segment_stuff(
                x=stuff_logits,
                y=stuff_gt,
                y_num=stuff_gt_num,
                y_mask=None,
                index_mask=stuff_gt_idx.reshape(-1),
            )
            loss_stuff = loss_stuff / max(stuff_gt_num, 1)
        else:
            loss_stuff = stuff_logits.sum() * 0.0

        result["loss_seg/stuff"] = loss_stuff

        # ============= #
        # Visualization #
        # ============= #

        # Perform visualization
        # if self.is_visualization_iteration:
        #     max_batch = 2
        #     images_dn = torch.vmap(self.denormalize_image)(images[:max_batch].detach())
        #     self.visualize(_V.visualize_true_things)(images_dn, true_thing, max_batch=max_batch)
        #     self.visualize(_V.visualize_true_stuff)(images_dn, true_stuff, max_batch=max_batch)
        #     self.visualize(_V.visualize_images)(images_dn)
        #     self.visualize(self.visualize_locations)(images_dn, multidets, gt_seg)

        #     if thing_gt_num > 0:
        #         self.visualize(_V.visualize_masks_thing)(
        #             things.logits,
        #             thing_gt,
        #             index_mask=thing_gt_idx.reshape(-1),
        #             weighted_num=self.weighted_num,
        #             weighted_values=thing_weights,
        #             instance_num=thing_num,
        #         )
        #     if stuff_gt_num > 0:
        #         self.visualize(_V.visualize_masks_stuff)(stuff.logits, stuff_gt, index_mask=stuff_gt_idx)

        # ================================ #
        # Position and localization losses #
        # ================================ #

        loss_location_ths, loss_location_sts = multi_apply(
            self.training_pipeline.losses_position, multidets, true_thing, true_stuff
        )
        result["loss_location/things"] = self.training_pipeline.loss_location_thing_weight * (
            sum(loss_location_ths) / max(thing_gt_num, 1)
        )
        result["loss_location/stuff"] = self.training_pipeline.loss_location_stuff_weight * (
            sum(loss_location_sts) / max(math.prod(inputs.batch_size), 2)
        )

        # ================= #
        # Depth computation #
        # ================= #

        # Depth losses
        if self.depth_mapper is None:
            return result

        depth_gt = self.training_pipeline.true_depths(inputs, ctx)
        depth_valid = depth_gt > 0.0

        # Filtering and selection for things
        select = [thing_gt[_idx][_s] * depth_valid[_idx] for _idx, _s in enumerate(thing_gt_idx)]
        keep = [s.sum(dim=(1, 2)) > 0 for s in select]
        select = [select[_idx][keep_s] for _idx, keep_s in enumerate(keep)]
        keep = torch.cat(keep)
        thing_mask = torch.zeros_like(thing_gt_idx).bool()
        temp = thing_mask[thing_gt_idx]
        temp[keep] = True
        thing_mask[thing_gt_idx] = temp
        del temp, keep

        if thing_mask.sum() == 0:
            merged_sum = (
                thing_kernels.get(self.depth_mapper.kernel_keys[0]).sum() * 0.00
                + ctx.embeddings.get(self.depth_mapper.feature_key).sum() * 0.0
            )
            result["loss_depth/thing/mean"] = merged_sum
            result["loss_depth/thing/value"] = merged_sum

            return result

        thing_dmap, thing_depth_mean = self.depth_mapper(features=ctx.embeddings, kernels=thing_kernels)

        # Mask out invalid pixels
        thing_dmap = einops.rearrange(thing_dmap, "b (nt nw) h w -> b nt nw h w", nw=self.weighted_num)
        thing_dmap = thing_dmap[thing_mask]

        # Compute depth loss for things
        thing_depth_losses = self.training_pipeline.losses_depth_thing(
            thing_dmap,
            thing_depth_mean,
            thing_mask,
            select,
            true_things=thing_gt,
            true_depths=depth_gt,
            weighted_num=self.weighted_num,
        )
        result.update(thing_depth_losses)

        # Generate depth map for stuff
        stuff_dmap, _ = self.depth_mapper(features=ctx.embeddings, kernels=stuff_kernels)

        # Compute depth loss for stuff
        result.update(
            self.training_pipeline.losses_depth_stuff(
                stuff_dmap,
                true_panseg=gt_seg,
                true_stuff=stuff_gt,
                true_stuff_idx=stuff_gt_idx,
                true_depths=depth_gt,
                depth_valid=depth_valid,
            )
        )
        return result

    # ----------------- #
    # Inference methods #
    # ----------------- #

    @torch.inference_mode()
    def _forward_inference(self, inputs: InputData, ctx: logic.Context) -> T.List[logic.training.TrainingResult]:
        """Implements the forward logic for inference, i.e. testing/evaluation mode."""

        # Process each image in the D2 ImageList
        processed_results = [self._forward_inference_single(_i, _c) for _i, _c in zip(inputs, ctx)]
        return processed_results

    @torch.inference_mode()
    def _forward_inference_single(self, inputs: InputData, ctx: logic.Context) -> logic.inference.InferenceResult:
        """Implements the forward logic for inference for a single item, that is not part of a batch."""

        res = {}

        # TODO merge inference pipe

        return res
