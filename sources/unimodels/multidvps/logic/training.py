"""Impleents the training pipeline."""

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override
from unipercept.modeling import losses
from uniutils.function import multi_apply, to_2tuple

from ._structures import Context, Labels, SampleInfo

if T.TYPE_CHECKING:
    from unipercept.data.points import InputData

    from ..modules import DepthHead, Detection
    from ..modules.supervision import Stuff, Things, TruthGenerator, Truths


__all__ = ["TrainingPipeline", "TrainingResult"]

TrainingResult: T.TypeAlias = T.Dict[str, torch.Tensor]  # losses


class TrainingPipeline(nn.Module):
    stuff_mask: T.Final[torch.Tensor]

    def __init__(
        self,
        *,
        loss_location_weight: float | T.Sequence[float] | tuple[float, float],  # things,stuff
        loss_location_thing: losses.SigmoidFocalLoss,
        loss_location_stuff: losses.SigmoidFocalLoss,
        loss_segment_thing: losses.WeightedThingDiceLoss,
        loss_segment_stuff: losses.WeightedStuffDiceLoss,
        loss_track_embedding: nn.TripletMarginWithDistanceLoss,
        loss_depth_values: nn.Module,
        loss_depth_means: nn.Module,
        loss_pgt: losses.PanopticGuidedTripletLoss,
        loss_dgp: losses.DepthGuidedPanopticLoss,
        truth_generator: TruthGenerator,
        stuff_channels: int,
    ):
        super().__init__()

        # Position loss
        self.loss_location_thing = loss_location_thing
        self.loss_location_stuff = loss_location_stuff
        self.loss_location_thing_weight, self.loss_location_stuff_weight = to_2tuple(loss_location_weight)

        assert self.loss_location_thing_weight >= 0.0 and math.isfinite(self.loss_location_thing_weight)
        assert self.loss_location_stuff_weight >= 0.0 and math.isfinite(self.loss_location_stuff_weight)

        # Segmentation loss
        self.loss_segment_things = loss_segment_thing
        self.loss_segment_stuff = loss_segment_stuff

        # Depth loss
        self.loss_depth_means = loss_depth_means
        self.loss_depth_values = loss_depth_values

        # PGT loss
        self.loss_pgt = loss_pgt
        self.loss_dgp = loss_dgp

        # Tracking loss
        self.loss_track_embedding = loss_track_embedding

        # Truth generator for training
        # self.truth_generator = torch.compile(
        #     truth_generator, dynamic=True, backend="eager", fullgraph=True, mode="reduce-overhead"
        # )
        self.truth_generator = truth_generator

        # Mask for semantic segmentation classes that are stuff
        stuff_mask = torch.ones(stuff_channels).bool()
        if T.TYPE_CHECKING:
            self.stuff_mask = stuff_mask
        else:
            self.register_buffer("stuff_mask", stuff_mask, False)

        if not self.truth_generator.stuff_all_classes:
            self.stuff_mask[-1] = False
        else:
            for id_embed in self.truth_generator.thing_embeddings.keys():
                self.stuff_mask[id_embed] = False

    @torch.jit.export
    @torch.inference_mode()
    @torch.no_grad()
    def true_segmentation(self, inputs: InputData, ctx: Context) -> Truths:
        assert inputs.captures.segmentations is not None

        hw_image: torch.Size = inputs.captures.images.shape[-2:]  # type: ignore
        hw_embedding: torch.Size = next(ctx.embeddings.values()).shape[-2:]
        hw_detections: T.Dict[str, torch.Size] = {key: d.shape[-2:] for key, d in ctx.detections.items()}

        gt = self.truth_generator.generate_panoptic(
            inputs.captures.segmentations,
            hw_image=hw_image,
            hw_embedding=hw_embedding,
            hw_detections=hw_detections,
        )

        return gt

    @torch.jit.export
    @torch.inference_mode()
    @torch.no_grad()
    def true_depths(self, inputs: InputData, ctx: Context) -> torch.Tensor:
        assert inputs.captures.depths is not None

        hw_embedding = next(ctx.embeddings.values()).shape[-2:]  # TODO: DRY

        gt: torch.Tensor = self.truth_generator.generate_depth(
            inputs.captures.depths,
            hw_embedding=hw_embedding,
        )

        return gt

    @torch.autocast("cuda", dtype=torch.float32)
    def losses_position(
        self, loc: Detection, true_thing: Things, true_stuff: Stuff
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the position loss for things and stuff. Applied on a single feature level.

        Results a tuple for use wih ``multi_apply``, as the final loss must be weighted later in the pipeline.
        """

        # Position loss for things
        loss_location_th = self.loss_location_thing(loc.thing_map, true_thing.scoremap.type_as(loc.thing_map))
        loss_location_th = loss_location_th.sum()

        # Position loss for stuff
        loss_location_st = self.loss_location_stuff(loc.stuff_map, true_stuff.scoremap.type_as(loc.stuff_map))
        loss_location_st = loss_location_st.mean((-2, -1))
        loss_location_st = loss_location_st.sum()

        return loss_location_th, loss_location_st

    # ---------------- #
    # Loss computation #
    # ---------------- #

    @torch.autocast("cuda", dtype=torch.float32)
    def losses_depth_thing(
        self,
        thing_dmap: torch.Tensor,
        thing_depth_mean: torch.Tensor,
        thing_mask: torch.Tensor,
        select: T.List[torch.Tensor],
        *,
        weighted_num: int,
        true_things: torch.Tensor,
        true_depths: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        result = {}

        h, w = true_things.shape[-2:]
        # Ground truth
        thing_depth_true = [_d.repeat((_s.shape[0], 1, 1)) for _d, _s in zip(true_depths, select)]
        thing_depth_true = torch.cat(thing_depth_true, dim=0).unsqueeze(1).type(thing_dmap.dtype)
        select_mask = torch.cat(select, dim=0).unsqueeze(1) > 0
        thing_depth_true = thing_depth_true * select_mask

        # Thing means
        if self.loss_depth_means is not None and self.thing_depth_mean.requires_grad:
            thing_depth_mean = rearrange(thing_depth_mean, "b (nt nw) () -> b nt nw () ()", nw=weighted_num)

            thing_depth_mean_true_raw = reduce(thing_depth_true, "nt () h w -> nt () () ()", "sum")
            thing_depth_mean_true_raw /= reduce(select_mask, "nt () h w -> nt () () ()", "sum").clamp(min=1)
            thing_depth_mean_true = torch.zeros_like(thing_depth_mean)
            thing_depth_mean_true[thing_mask] = thing_depth_mean_true_raw.type_as(thing_depth_mean)
            del thing_depth_mean_true_raw

            thing_mask = thing_mask.repeat((weighted_num, 1, 1)).permute(1, 2, 0)
            thing_depth_mean_true = thing_depth_mean_true.reshape(thing_mask.shape[0], 1, -1)
            thing_depth_mean = thing_depth_mean.reshape(thing_mask.shape[0], 1, -1)
            thing_mask = thing_mask.reshape(thing_mask.shape[0], 1, -1)

            result["loss_depth/thing/mean"] = self.loss_depth_means(
                thing_depth_mean_true,
                thing_depth_mean,
                mask=thing_mask,
            )

        # Repeat things depth GT for the amount of weighted instances
        thing_depth_true = thing_depth_true.repeat(1, weighted_num, 1, 1)

        result["loss_depth/thing/value"] = self.loss_depth_values(
            thing_depth_true,
            thing_dmap,
            mask=thing_depth_true > 0,
        )

        return result

    @torch.autocast("cuda", dtype=torch.float32)
    def losses_depth_stuff(
        self,
        stuff_dmap: torch.Tensor,
        *,
        true_panseg: Truths,
        true_stuff: torch.Tensor,
        true_stuff_idx: torch.Tensor,
        true_depths: torch.Tensor,
        depth_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = {}

        true_panseg.type_as_(true_stuff)

        gt_sem_one_hot = [gt.scoremap.sum(dim=(-1, -2)) > 0 for gt in true_panseg.stuff.values()]
        stuff_mask = [stuff_mask.unsqueeze(0) & s for s in gt_sem_one_hot]
        gt_sem_one_hot = torch.cat(gt_sem_one_hot, dim=1)
        stuff_mask = torch.cat(self.stuff_mask, dim=1)

        select_st = [true_stuff[_idx][select_st] * depth_valid[_idx] for _idx, select_st in enumerate(true_stuff_idx)]
        keep = [(s.sum(dim=(1, 2)) > 0) & sm[gs > 0] for s, sm, gs in zip(select_st, stuff_mask, gt_sem_one_hot)]
        select_st = [select_st[_idx][keep_s] for _idx, keep_s in enumerate(keep)]

        # Compute overall depth loss
        stuff_dmap = stuff_dmap[true_stuff_idx][torch.cat(keep)]
        stuff_depth_true = [d.repeat((s.shape[0], 1, 1)) for d, s in zip(true_depths, select_st)]
        stuff_depth_true = torch.cat(stuff_depth_true, dim=0).unsqueeze(1).type_as(stuff_dmap)

        # Concatenate selection
        select_st = torch.cat(select_st, dim=0).unsqueeze(1) > 0.5

        # Concatenate stuff and thing depths to compute overall depth
        stuff_depth_true = stuff_depth_true * select_st

        result["loss_depth/stuff/value"] = self.loss_depth_values(
            stuff_depth_true,
            stuff_dmap.unsqueeze(1),
            mask=stuff_depth_true > 0,
        )

        return result

    @override
    def forward(self, inputs: InputData) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Pipeline is not callable directly.")


# --------------------- #
# Thing/Stuff Detection #
# --------------------- #


def detect_things(
    det: Detection, true_ths: Things, weighted_num: int, *, ignored_label=-1
) -> tuple[TensorDictBase, int, torch.Tensor]:
    if true_ths.num_instances == 0:
        empty_kernels = det.kernel_spaces.apply(
            lambda t: torch.empty((t.shape[0], 0, t.shape[1]), device=t.device, dtype=t.dtype),
            batch_size=((det.batch_size[0], 0)),
        )
        empty_weights = torch.zeros(
            (det.batch_size[0], 0, weighted_num), device=det.thing_map.device, dtype=torch.float
        )

        return empty_kernels, 0, empty_weights

    true_th = true_ths.instances
    # TODO
    # Downscale instance masks to the size of the feature map

    with torch.no_grad():
        guided_inst = nn.functional.interpolate(
            true_th.insts,
            size=det.hw,
            mode="bilinear",
            align_corners=False,
        ).type_as(det.thing_map)
        keep = (guided_inst > 1e-2) & (guided_inst < ignored_label)
        guidence = torch.zeros_like(guided_inst)

    # For each batch item, select the score maps that correspond to the categories
    # in the ground truth
    pred_select = torch.vmap(torch.index_select, in_dims=(0, None, 0))(
        det.thing_map, 0, true_th.categories * true_th.indices_mask
    )
    guidence[keep] = pred_select[keep].sigmoid()
    # print("Test:", self.weighted_num)
    guidence_reshaped = guidence.reshape(guided_inst.shape[0], guided_inst.shape[1], -1)
    corrected_weighted_num = min(guidence_reshaped.shape[2], weighted_num)

    weighted_values, guided_index = torch.topk(
        guidence_reshaped,
        k=corrected_weighted_num,
        # k=self.weighted_num,
        dim=-1,
    )

    thing_num = max(1, int(true_th.indices_mask.sum(dim=1).max()))
    thing_total = thing_num * corrected_weighted_num

    guided_index = guided_index[:, :thing_num, :]
    guided_index = rearrange(guided_index, "batch thing_num weighted_num -> batch (thing_num weighted_num)")
    weighted_values = weighted_values[:, :thing_num, :]

    def padding_weighted_num(k_space: torch.Tensor) -> torch.Tensor:
        k_space = rearrange(
            k_space,
            "batch (thing_num weighted_corr) dims -> batch thing_num weighted_corr dims",
            weighted_corr=corrected_weighted_num,
        )

        # TODO: Padding to shape: "batch thing_num weighted_num dims"
        padding = (0, 0, 0, weighted_num - corrected_weighted_num)  # Verschil padding links of rechts?
        k_space = nn.functional.pad(k_space, padding, "constant", 0)

        # TODO shape back to "batch (thing_num weighted_num) dims"
        k_space = rearrange(k_space, "batch thing_num weighted_self dims -> batch (thing_num weighted_self) dims")

        return k_space

    def sample_thing_kernels(k_space: torch.Tensor) -> torch.Tensor:
        # TODO: Possible performance gains by using torch builtins directly?
        k_space = rearrange(k_space, "batch dims h w -> batch dims (h w)")
        i = repeat(guided_index, "batch num -> batch dims num", dims=k_space.shape[1])
        k = torch.gather(k_space, dim=2, index=i)
        k = rearrange(k, "batch dims num -> batch num dims")

        assert k.shape[1] == thing_total, f"{k.shape[1]} != {thing_total} != {thing_num} * {corrected_weighted_num}"

        return k

    thing_kernels = det.kernel_spaces.apply(sample_thing_kernels, batch_size=(det.shape[0], thing_total))
    assert thing_kernels.shape[:2] == (det.batch_size[0], thing_total)
    # TODO thing_kernels terug shapen zodat je hem kan padden
    # Current  shape: "batch (thing_num weighted_corr) dims"
    # Desired shape: "batch thing_num weighted_corr dims"
    thing_kernels = thing_kernels.apply(padding_weighted_num, batch_size=(det.shape[0], thing_num * weighted_num))

    padding = (0, weighted_num - corrected_weighted_num)
    weighted_values = nn.functional.pad(weighted_values, padding, "constant", 0)

    return thing_kernels, thing_num, weighted_values


def detect_stuff(
    det: Detection,
    true_st: Stuff,
) -> T.Tuple[TensorDictBase, int]:
    stuff_num = max(1, true_st.num_instances)
    stuff_masks = true_st.masks[:, :stuff_num]
    stuff_pixels = reduce(stuff_masks, "batch num h w -> batch num ()", "sum").clamp(1.0)

    def sample_stuff_kernels(k_space: torch.Tensor) -> torch.Tensor:
        # TODO: Possible performance gains by using torch builtins directly?
        k = einsum(stuff_masks, k_space, "batch num h w, batch dims h w -> batch num dims h w")
        k = reduce(k, "batch num dims h w -> batch num dims", "sum")
        k = k / stuff_pixels

        assert k.shape[1] == stuff_num, f"{k.shape[1]} != {stuff_num}"

        return k

    stuff_kernels = det.kernel_spaces.apply(sample_stuff_kernels, batch_size=(det.shape[0], stuff_num))

    assert stuff_kernels.shape[:2] == (det.shape[0], stuff_num)  # type: ignore

    return stuff_kernels, stuff_num
