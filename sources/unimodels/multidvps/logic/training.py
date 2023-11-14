"""Impleents the training pipeline."""

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override
from unimodels.multidvps.keys import KEY_REID

from unipercept.nn import losses
from unipercept.utils.function import multi_apply, to_2tuple

from ._structures import Context

if T.TYPE_CHECKING:
    from unipercept.model import InputData

    from ..modules import DepthHead, Detection
    from ..modules.supervision import Stuff, Things, TruthGenerator, Truths


__all__ = ["TrainingPipeline", "TrainingResult"]

TrainingResult: T.TypeAlias = T.Dict[str, torch.Tensor]  # losses


class ThingDepthLosses(T.TypedDict):
    depth_thing_mean: torch.Tensor
    depth_thing_value: torch.Tensor


class StuffDepthLosses(T.TypedDict):
    depth_stuff_value: torch.Tensor


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
        loss_reid: nn.TripletMarginLoss,
        loss_depth_values: nn.Module,
        loss_depth_means: nn.Module,
        loss_pgt: losses.PGTLoss,
        loss_dgp: losses.DGPLoss,
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
        self.loss_reid: nn.TripletMarginWithDistanceLoss = loss_reid  # type: ignore

        # Truth generator for training
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

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        """
        Initialize from a dataset/datainfo name using the Metadata object.
        """
        from unipercept.data.sets import Metadata, get_info

        info: Metadata = get_info(name)
        return cls(stuff_channels=info.stuff_amount, **kwargs)

    @torch.jit.export
    def losses_tracking(
        self,
        kernels: torch.Tensor,
        weights: torch.Tensor,
        *,
        index_mask: torch.Tensor,
        labels: torch.Tensor,
        instance_num: int,
        weight_num: int,
    ) -> torch.Tensor:
        """
        Computes the tracking embedding loss using a TripletMarginWithDistanceLoss.

        Parameters
        ----------
        kernels
            The kernels to compute the loss on.
        weights
            The instance weights for each kernel.
        index_mask
            A mask that indices which kernels have a valid ground turth instance.
        labels
            The ground truth labels for each kernel.
        instance_num
            The number of instances per image.
        weight_num
            The number of weighted instances per image.
        """
        batch_size, _, dims = kernels.shape  # n, instance_num * weighted_num, kernel_dims

        # Create the lists of anchors, positive and negative examples.
        # The anchor mask is a boolean mask indicating which kernels are a valid anchor
        # The positive and negative indices are the indices of the positive and negative examples for each anchor
        with torch.no_grad():
            labels = rearrange(labels, "(b p) n c -> b (p n) c", p=2)
            categories = labels[..., 0]
            ids = labels[..., 1]
            valid_mask = index_mask.reshape_as(ids)

            # Create triplets
            anc_mask, pos_idx, neg_idx = _create_instance_triplets(ids, categories, valid_mask)

            # Translate the indices such that they can later index the embeddings tensor at the correct position
            # idx_flat = torch.arange(batch_size * instance_num, device=ids.device).reshape(batch_size, instance_num)
            # idx_flat[~index_mask] = -1
            # idx_flat = idx_flat.reshape_as(anc_mask)

            mask_flat = index_mask.flatten()
            idx_flat = torch.zeros(batch_size * instance_num, device=ids.device, dtype=torch.float)
            idx_flat[mask_flat] = 1.0
            idx_flat.cumsum_(0).sub_(1.0)
            idx_flat[~mask_flat] = -1.0
            idx_flat = idx_flat.long().reshape_as(anc_mask)

            anc_idx = idx_flat[anc_mask]
            pos_idx = torch.gather(idx_flat, dim=-1, index=pos_idx)[anc_mask]
            neg_idx = torch.gather(idx_flat, dim=-1, index=neg_idx)[anc_mask]

            assert (
                anc_idx.shape == pos_idx.shape == neg_idx.shape
            ), f"Shapes of index tensors do not match: {anc_idx.shape} != {pos_idx.shape} != {neg_idx.shape}"
            assert torch.all(anc_idx >= 0), "Invalid anchor indices"
            assert torch.all(pos_idx >= 0), "Invalid positive indices"
            assert torch.all(neg_idx >= 0), "Invalid negative indices"

        # Reshape the kernels and weights to [batch_size, instance_num, weighted_num, dims] and then select the
        # valid entires using the index mask
        kernels = kernels.reshape(batch_size, instance_num, weight_num, dims)
        kernels = kernels[index_mask, ...]

        weights = weights.float()
        weights = weights[index_mask, ...]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(1e-6)

        # Compute the weighted kernels as the weighted sum of kernels [n, weight_num, dims] and weights [n, weight_num]
        embeddings = einsum(kernels, weights, "m w d, m w -> m d")

        # Compute the triplet loss
        anc = embeddings[anc_idx]
        pos = embeddings[pos_idx]
        neg = embeddings[neg_idx]

        triplet_loss = self.loss_reid(anc, pos, neg)

        return triplet_loss

    @torch.jit.export
    @torch.no_grad()
    def true_segmentation(self, ctx: Context) -> Truths:
        assert ctx.captures.segmentations is not None

        hw_image: torch.Size = ctx.captures.images.shape[-2:]  # type: ignore
        hw_embedding: torch.Size = next(e for e in ctx.embeddings.values()).shape[-2:]
        hw_detections: T.Dict[str, torch.Size] = {key: d.stuff_map.shape[-2:] for key, d in ctx.detections.items()}

        gt = self.truth_generator.generate_panoptic(
            ctx.captures.segmentations,
            hw_image=hw_image,
            hw_embedding=hw_embedding,
            hw_detections=hw_detections,
        )

        return gt

    @torch.jit.export
    # @torch.inference_mode()
    @torch.no_grad()
    def true_depths(self, ctx: Context) -> torch.Tensor:
        assert ctx.captures.depths is not None

        hw_embedding = next(e for e in ctx.embeddings.values()).shape[-2:]  # TODO: DRY

        gt: torch.Tensor = self.truth_generator.generate_depth(
            ctx.captures.depths,
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
    ) -> ThingDepthLosses:
        result = {}

        h, w = true_things.shape[-2:]
        # Ground truth
        thing_depth_true = [_d.repeat((_s.shape[0], 1, 1)) for _d, _s in zip(true_depths, select)]
        thing_depth_true = torch.cat(thing_depth_true, dim=0).unsqueeze(1).type(thing_dmap.dtype)
        select_mask = torch.cat(select, dim=0).unsqueeze(1) > 0
        thing_depth_true = thing_depth_true * select_mask

        # Thing means
        if self.loss_depth_means is not None and thing_depth_mean.requires_grad:
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

            result["depth_thing_mean"] = self.loss_depth_means(
                thing_depth_mean_true,
                thing_depth_mean,
                mask=thing_mask,
            )

        # Repeat things depth GT for the amount of weighted instances
        thing_depth_true = thing_depth_true.repeat(1, weighted_num, 1, 1)

        result["depth_thing_value"] = self.loss_depth_values(
            thing_depth_true,
            thing_dmap,
            mask=thing_depth_true > 0,
        )

        return result  # type: ignore

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
        stuff_mask = [self.stuff_mask.unsqueeze(0) & s for s in gt_sem_one_hot]
        gt_sem_one_hot = torch.cat(gt_sem_one_hot, dim=1)
        stuff_mask = torch.cat(stuff_mask, dim=1)

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

        result["depth_stuff"] = self.loss_depth_values(
            stuff_depth_true,
            stuff_dmap.unsqueeze(1),
            mask=stuff_depth_true > 0,
        )

        return result

    @override
    def forward(self, ctx: Context, spec: torch.Size) -> dict[str, torch.Tensor]:
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
        keep = (guided_inst > 1e-2) & (guided_inst > ignored_label)
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


_multinomial_batched = torch.vmap(torch.multinomial, in_dims=(0, None, None), out_dims=0, randomness="different")


@torch.no_grad()
def _create_instance_triplets(
    ids: torch.Tensor, categories: torch.Tensor, valid_mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates triplets of anchor, positive and negative examples for each instance in the batch.

    NOTE: Returns randomised postive and negative examples for each index that is not an anchor (e.g. entries that are
    false in the anchor mask).

    Parameters
    ----------
    ids
        The instance IDs for each kernel.
    categories
        The semantic categories for each kernel.

    Returns
    -------
    anchor_mask
        A boolean mask indicating which kernels are anchors.
    (positive_indices, negative_indices)
        The indices of the positive and negative examples for each anchor.
    """

    assert ids.shape == categories.shape, f"Invalid shape for ids and categories: {ids.shape} != {categories.shape}"
    assert ids.ndim >= 2, "ids must be a 2d tensor (batch, ids)"

    # Create the matrices of equal and not equal pairs with respect to the their ids
    ids_eq = ids.unsqueeze(-2) == ids.unsqueeze(-1)
    ids_ne = ~ids_eq

    # Create a validity indicating mask
    if valid_mask is None:
        valid_mask = torch.ones_like(ids_eq, dtype=torch.float)
    else:
        valid_mask = (valid_mask.unsqueeze(-2) & valid_mask.unsqueeze(-1)).float()

    # Convert to floating point such that we can sum the indices and use them as weights in a multinomial distribution
    ids_eq = ids_eq.float()
    ids_eq.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    ids_ne = ids_ne.float()

    # Mask out invalid entries
    ids_eq *= valid_mask
    ids_ne *= valid_mask

    ids_eq_count = ids_eq.sum(dim=-1)
    ids_ne_count = ids_ne.sum(dim=-1)

    # Check whether it can be paired with some positive and negative example
    anchor_mask = (ids_eq_count >= 1) & (ids_ne_count >= 1)

    # We will draw indices from a multinomial distribution with the weights being the float-valued indices of the
    # positive and negative examples.
    # For the cases where no triplet is possible, add a dummy '1' in the ids_eq and ids_ne matrices such that we can
    # still draw a sample from the multinomial distribution without having to index, which breaks the vmap.
    dummy_weights = (~anchor_mask).unsqueeze(-1).float()
    ids_eq += dummy_weights
    ids_ne += dummy_weights

    # We want to additionally add some weight to the negative examples that have the same category as the anchor.
    cat_eq = (categories.unsqueeze(-2) == categories.unsqueeze(-1)).float()
    cat_eq.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    cat_eq *= valid_mask

    ids_ne += cat_eq

    # For each anchor, we randomly sample a positive and negative example using `multinomial` with the weights being
    # the float-valued indices of the positive and negative examples.
    positive_indices = _multinomial_batched(ids_eq, 1, True).squeeze_(-1)
    negative_indices = _multinomial_batched(ids_ne, 1, True).squeeze_(-1)

    return anchor_mask, positive_indices, negative_indices
