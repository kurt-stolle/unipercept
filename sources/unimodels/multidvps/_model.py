"""Implements the main MultiDVPS model class."""

import math
import typing as T
from calendar import c
import warnings

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override
from unimodels.multidvps import logic, modules
from unimodels.multidvps.keys import (
    KEY_DEPTH,
    KEY_MASK,
    KEY_REID,
    OUT_BACKGROUND,
    OUT_DEPTH,
    OUT_OBJECT,
    OUT_PANOPTIC,
)

from unipercept.utils.function import multi_apply

from unimodels.multidvps.modules import (
    DepthHead,
    Detector,
    FeatureEncoder,
    MaskHead,
    StuffFusion,
    ThingFusion,
    KernelMapper,
)
from unipercept.nn.backbones import Backbone
from unipercept.nn.layers.tracking import StatefulTracker

import unipercept as up

__all__ = ["MultiDVPS"]


_M = T.TypeVar("_M", bound=nn.Module)

def _maybe_optimize_submodule(module: _M, **kwargs) -> _M:
    try:
        module = T.cast(_M, torch.compile(module, **kwargs))
    except Exception as err:
        warnings.warn(f"Could not compile submodule {module.__class__.__name__}: {err}")
    return module

class MultiDVPS(up.model.ModelBase):
    """Depth-Aware Video Panoptic Segmentation model using dynamic convolutions."""

    id_map_thing: T.Final[T.Dict[int, int]]
    id_map_stuff: T.Final[T.Dict[int, int]]
    stuff_with_things: T.Final[bool]
    stuff_all_classes: T.Final[bool]
    depth_fixed: T.Final[T.Dict[int, float]]
    thing_embeddings: T.Final[T.List[str]]
    stuff_embeddings: T.Final[T.List[str]]

    def __init__(
        self,
        *,
        common_stride: int,
        weighted_num: int,
        backbone: Backbone,
        detector: Detector,
        kernel_mapper: KernelMapper,
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
        depth_mapper: T.Optional[DepthHead],
        thing_embeddings: T.List[str],
        stuff_embeddings: T.List[str],
        depth_fixed: T.Optional[dict[int, float]] = None,
        tracker: T.Optional[StatefulTracker] = None,
    ) -> None:
        super().__init__()

        # Properties
        self.weighted_num = weighted_num
        self.common_stride = common_stride

        # Metadata readable from the dataset or passed as literal
        self.id_map_thing = {int(k): int(v) for k, v in id_map_thing.items()}
        self.id_map_stuff = {int(k): int(v) for k, v in id_map_stuff.items()}
        self.stuff_with_things = stuff_with_things
        self.stuff_all_classes = stuff_all_classes

        # Keys that are used for the thing and stuff embeddings
        self.thing_embeddings = thing_embeddings
        self.stuff_embeddings = stuff_embeddings

        # Categories for which the depth always has a fixed value, e.g. the sky.
        self.depth_fixed = {} if depth_fixed is None else depth_fixed

        # Submodules
        self.backbone = _maybe_optimize_submodule(backbone)
        self.detector = detector
        self.kernel_mapper = kernel_mapper
        self.fusion_thing = fusion_thing
        self.fusion_stuff = fusion_stuff
        self.feature_encoder = feature_encoder
        self.maskifier_thing = maskifier_thing
        self.maskifier_stuff = maskifier_stuff
        self.depth_mapper = depth_mapper
        self.tracker = tracker

        # Pipelines
        self.training_pipeline = training_pipeline
        self.inference_pipeline = inference_pipeline

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> T.Self:
        info = up.data.sets.get_info(dataset_name)

        return cls(
            id_map_stuff=info.stuff_train_id2contiguous_id,
            id_map_thing=info.thing_train_id2contiguous_id,
            stuff_all_classes=info.stuff_all_classes,
            stuff_with_things=info.stuff_with_things,
            depth_fixed=info.depth_fixed,
            **kwargs,
        )

    @override
    def forward(self, inputs: up.model.InputData) -> up.model.ModelOutput:
        """
        Implements the forward logic for the full model.

        Backbone feature extraction, detection/kernel generation, and high-resolution feature embedding generation
        are shared across training and inference.

        Further processing is split into training and inference branches, see ``_forward_training`` and
        ``_forward_inference``.
        """

        # Common backbone, feature extraction and detection
        ctx = self._forward_common(inputs)

        # Return losses (training) or predictions (inference)
        if self.training:
            return self._forward_training(inputs, ctx)
        else:
            return self._forward_inference(inputs, ctx)

    def _forward_common(self, inputs: up.model.InputData) -> logic.Context:
        captures_flat = inputs.captures.flatten()  # .contiguous()
        features = self.backbone(captures_flat.images)
        highres = self.feature_encoder(features)
        detections = self.detector(features)

        # Create combined objects context
        return logic.Context(
            captures=captures_flat,
            detections=detections,
            embeddings=highres,
            batch_size=captures_flat.batch_size,
            device=captures_flat.device,
        )

    def _map_thing_kernels(self, kernels: TensorDict) -> TensorDict:
        return self.kernel_mapper(kernels, self.thing_embeddings)

    def _map_stuff_kernels(self, kernels: TensorDict) -> TensorDict:
        return self.kernel_mapper(kernels, self.stuff_embeddings)

    # ---------------- #
    # Training methods #
    # ---------------- #
    def _forward_training(self, inputs: up.model.InputData, ctx: logic.Context) -> up.model.ModelOutput:
        """Implements the forward logic for training."""

        # Allocate outputs
        outputs = up.model.ModelOutput(batch_size=[])

        # Generate ground truth segmentation
        gt_seg = self.training_pipeline.true_segmentation(ctx)
        multidets, true_thing, true_stuff = tuple(
            zip(*[(ctx.detections[key], gt_seg.thing[key], gt_seg.stuff[key]) for key in self.detector.in_features])
        )

        # ============= #
        # Kernel fusion #
        # ============= #

        # Thing generation
        thing_kernels_multi, thing_nums, thing_weights = multi_apply(
            logic.training.detect_things, multidets, true_thing, weighted_num=self.weighted_num
        )
        thing_kernels: TensorDict = torch.cat(thing_kernels_multi, dim=1)  # type: ignore
        thing_kernels = self._map_thing_kernels(thing_kernels)
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
        stuff_kernels = self._map_stuff_kernels(stuff_kernels)
        stuff_kernels, _, _ = self.fusion_stuff(stuff_kernels, None, None)

        # ================== #
        # Thing segmentation #
        # ================== #
        (
            thing_gt,
            thing_gt_idx,
            thing_gt_lbl,
            thing_gt_num,
        ) = gt_seg.mask_instances({k: n for k, n in zip(list(self.detector.in_features), thing_nums)})

        # Compute segmentation loss
        thing_logits = self.maskifier_thing(
            features=ctx.embeddings,
            kernels=thing_kernels,
        )

        if thing_gt_num > 0:
            loss_thing = self.training_pipeline.loss_segment_things(
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
            loss_thing = thing_logits.mean() * 0.0

        outputs.truths["thing_mask"] = thing_gt.detach()
        if thing_num > 0:
            outputs.predictions["thing_mask"] = thing_logits.detach().sigmoid_().argmax(dim=1)
        outputs.losses["segmentation.things"] = loss_thing

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

        outputs.truths["stuff_mask"] = stuff_gt.detach()
        outputs.predictions["stuff_mask"] = stuff_logits.detach().sigmoid_().argmax(dim=1)
        outputs.losses["segmentation.stuff"] = loss_stuff

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
        outputs.losses["position.things"] = self.training_pipeline.loss_location_thing_weight * (
            sum(loss_location_ths) / max(thing_gt_num, 1)
        )
        outputs.losses["position.stuff"] = self.training_pipeline.loss_location_stuff_weight * (
            sum(loss_location_sts) / max(math.prod(ctx.batch_size), 1)
        )

        # ============= #
        # Tracking loss #
        # ============= #

        if self.tracker is not None and thing_gt_num > 0:
            loss_track = self.training_pipeline.losses_tracking(
                thing_kernels.get(KEY_REID),
                thing_weights,  # type: ignore
                index_mask=thing_gt_idx,
                labels=thing_gt_lbl,
                instance_num=thing_num,
                weight_num=self.weighted_num,
            )
            outputs.losses["track"] = loss_track
        else:
            # NOTE: This is a dummy loss to keep the reid-kernel in the graph
            outputs.losses["track"] = thing_kernels[KEY_REID].sum() * 0.0

        # ============ #
        # Depth losses #
        # ============ #

        if self.depth_mapper is not None:
            depth_gt = self.training_pipeline.true_depths(ctx)
            depth_valid = depth_gt > 0.0

            outputs.losses["guided.pgt"] = self.training_pipeline.loss_pgt(
                ctx.embeddings.get(KEY_DEPTH), ctx.captures.segmentations
            )
            outputs.losses["guided.dgp"] = self.training_pipeline.loss_dgp(ctx.embeddings.get(KEY_MASK), depth_gt)

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

            thing_depth_losses: logic.training.ThingDepthLosses
            if thing_mask.sum() == 0:
                merged_sum = (
                    thing_kernels.get(self.depth_mapper.geometry_key).sum() * 0.0
                    + thing_kernels.get(self.depth_mapper.feature_key).sum() * 0.0
                    + ctx.embeddings.get(self.depth_mapper.feature_key).sum() * 0.0
                )

                thing_depth_losses = {"depth_thing_mean": merged_sum, "depth_thing_value": merged_sum}
            else:
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
            outputs.losses.update(thing_depth_losses)

            # Generate depth map for stuff
            stuff_dmap, _ = self.depth_mapper(features=ctx.embeddings, kernels=stuff_kernels, return_means=False)

            # Compute depth loss for stuff
            outputs.losses.update(
                self.training_pipeline.losses_depth_stuff(
                    stuff_dmap,
                    true_panseg=gt_seg,
                    true_stuff=stuff_gt,
                    true_stuff_idx=stuff_gt_idx,
                    true_depths=depth_gt,
                    depth_valid=depth_valid,
                )
            )
        return outputs  # type: ignore

    def visualize_training(self, outputs: up.model.ModelOutput) -> dict[str, torch.Tensor]:
        results = {}
        true_thing_mask = outputs.get(("truths", "thing_mask"), None)
        if true_thing_mask is not None:
            results["thing_mask"] = true_thing_mask

        return results

    # ----------------- #
    # Inference methods #
    # ----------------- #

    def _forward_inference(self, inputs: up.model.InputData, ctx: logic.Context) -> up.model.ModelOutput:
        """Implements the forward logic for inference, i.e. testing/evaluation mode."""

        # Unflatten captures
        ctx = ctx.unflatten(0, inputs.captures.batch_size)

        # Results dictionary
        outputs = []

        # Process each image
        for i in range(0, inputs.batch_size[0]):
            inputs_batch = inputs[0]
            ctx_batch = ctx[i : i + 1].flatten()

            # Upscale depth feature
            if KEY_DEPTH in ctx_batch.embeddings.keys():
                feat_depth = ctx_batch.embeddings.get(KEY_DEPTH)
                feat_depth = self.inference_pipeline.upscale_to_input_size(ctx_batch, feat_depth)

                ctx_batch.embeddings = ctx_batch.embeddings.set(KEY_DEPTH, feat_depth)

            # Predict depth-aware panoptic segmentation
            outputs_batch = self._predict(inputs_batch, ctx_batch)  # .contiguous()
            outputs.append(outputs_batch)

        return torch.stack(outputs)  # type: ignore

    def _predict(self, inputs: up.model.InputData, ctx: logic.Context) -> up.model.ModelOutput:
        assert len(ctx.batch_size) == 1, "Inference pipeline accepts only single samples"
        assert len(inputs.batch_size) == 0, "Inference pipeline accepts only single samples"
        assert (
            len(inputs.captures.batch_size) == 1 and inputs.captures.batch_size[0] == 1
        ), "Inference pipeline accepts unpaired samples"

        outputs = up.model.ModelOutput(batch_size=[])

        # Infer individual object categoires (things)
        things = self._predict_things(ctx)
        stuff = self._predict_stuff(ctx)

        # Combine things and stuff
        panoptic, depth, things = self.inference_pipeline.merge_predictions(
            ctx,
            things,
            stuff,
            stuff_all_classes=self.stuff_all_classes,
            stuff_with_things=self.stuff_with_things,
            id_map_thing=self.id_map_thing,
            id_map_stuff=self.id_map_stuff,
        )

        # Store outputs
        outputs.predictions[OUT_BACKGROUND] = stuff
        outputs.predictions[OUT_OBJECT] = things
        outputs.predictions[OUT_PANOPTIC] = panoptic
        if inputs.captures.segmentations is not None:
            outputs.truths[OUT_PANOPTIC] = inputs.captures.segmentations[-1, :, :]  # select last entry in pair

        outputs.predictions[OUT_DEPTH] = depth
        if inputs.captures.depths is not None:
            outputs.truths[OUT_DEPTH] = inputs.captures.depths[-1, :, :]

        if things.num_instances > 0:
            self._apply_tracking_(inputs, outputs)

        return outputs

    def _apply_tracking_(self, inputs: up.model.InputData, outputs: up.model.ModelOutput) -> None:
        ins = outputs.predictions[OUT_OBJECT]

        # Skip if no instances were detected
        if ins.num_instances == 0:
            return

        # Skip if tracking is disabled
        if self.tracker is None:
            num = ins.batch_size[-1]
            ins.set_(
                "iids",
                torch.tensor(
                    list(range(num)),
                    device=inputs.device,
                )
                + 1,
            )
            return

        # Create inputs to the tracking algorithm
        tracker_inputs = TensorDict(
            {
                "inputs": inputs,
                "outputs": outputs,
            },
            batch_size=[],
        )

        # Actual tracking
        tracking_ids = self.tracker(tracker_inputs, key=inputs.ids[0], frame=inputs.ids[1])

        # Update the instance ids in the class
        ins.set_("iids", tracking_ids)

        # Update the instance ids in the panoptic segmentation map
        sem_map, ins_map_detect = (
            outputs.predictions[OUT_PANOPTIC].as_subclass(up.data.tensors.PanopticMap).to_parts(as_tuple=True)
        )
        ins_map_tracked = torch.zeros_like(ins_map_detect)

        for idx, id_ in enumerate(ins.iids):
            idx = idx + 1
            ins_map_tracked[ins_map_detect == idx] = id_.type_as(ins_map_detect)

        outputs.predictions.set_(OUT_PANOPTIC, up.data.tensors.PanopticMap.from_parts(sem_map, ins_map_tracked))

    def _predict_things(self, ctx: logic.Context) -> logic.ThingInstances:
        kernels, cats, scores = self.inference_pipeline.predict_things(ctx)

        # Generate things
        kernels = self._map_thing_kernels(kernels)
        kernels, cats, scores = self.fusion_thing(kernels, cats, scores)
        thing_logits = self.maskifier_thing(
            features=ctx.embeddings,
            kernels=kernels,
        ).sigmoid_()

        # Generate depth for detected instances
        if self.depth_mapper is not None:
            dmap, means = self.depth_mapper(features=ctx.embeddings, kernels=kernels)
            depth = modules.DepthPrediction(
                maps=dmap, means=means, batch_size=kernels.batch_size, device=None
            )  # .view(-1)
        else:
            depth = None

        masks = thing_logits > self.inference_pipeline.inst_thres

        score_factor = (thing_logits * masks.float()).sum((-2, -1))
        score_factor /= masks.sum((-2, -1)).float().clamp(1e-8)
        scores *= score_factor

        things = logic.ThingInstances(
            kernels=kernels,
            logits=thing_logits,
            masks=masks,
            scores=scores,
            categories=cats,
            iids=torch.zeros_like(cats),
            depths=depth,
            batch_size=kernels.batch_size,
        ).flatten()

        # Sorting index by score
        sort_index = torch.argsort(things.scores, descending=True)
        # things.apply(lambda x: torch.index_select(x, 0, sort_index), inplace=True)
        things = things[sort_index]

        # Keep only scores above a threshold
        keep_mask = things.scores >= 0.05
        if not keep_mask.any():
            return things[:0]

        things = things.masked_select(keep_mask)

        # Sort again and keep only the top instances
        sort_index = torch.argsort(things.scores, descending=True)
        sort_index = sort_index[: self.inference_pipeline.center_top_num]
        things = things[sort_index]

        return things  # .contiguous()

    def _predict_stuff(self, ctx: logic.Context) -> logic.StuffInstances:
        kernels, categories, scores = self.inference_pipeline.predict_stuff(
            ctx, stuff_all_classes=self.stuff_all_classes, stuff_with_things=self.stuff_with_things
        )

        # Fusion
        kernels = self._map_stuff_kernels(kernels)
        kernels, categories, scores = self.fusion_stuff(kernels, categories, scores)

        # Generate semantic predictions
        stuff_logits = self.maskifier_stuff(
            features=ctx.embeddings,
            kernels=kernels,
        ).sigmoid_()

        # Generate depth for detected instances
        if self.depth_mapper is not None:
            dmap, means = self.depth_mapper(features=ctx.embeddings, kernels=kernels)
            assert dmap is not None
            depth = modules.DepthPrediction(maps=dmap, means=means, batch_size=kernels.batch_size)
        else:
            depth = None

        # Convert to results object, which requires removal of the fake batch dimension
        stuff = logic.StuffInstances(
            kernels=kernels,
            scores=scores,
            categories=categories,
            logits=stuff_logits,
            depths=depth,
            batch_size=categories.shape,
            device=ctx.device,
        ).flatten()

        return stuff  # .contiguous()
