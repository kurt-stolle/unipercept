"""
Various visualization utilties for MultiDVPS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from .modules import supervision


def visualize_locations(
    img: torch.Tensor,
    *,
    gt_center: torch.Tensor,
    gt_region: torch.Tensor,
    pr_center: torch.Tensor | None = None,
    pr_region: torch.Tensor | None = None,
    max_rows=2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Yields tensor visualization for things and stuff as a tuple
    """

    if pr_center is None:
        pr_center = torch.zeros_like(gt_center)
    if pr_region is None:
        pr_region = torch.zeros_like(gt_region)

    img, pr_center, pr_region, gt_center, gt_region = map(
        lambda t: t.detach().cpu(), (img, pr_center, pr_region, gt_center, gt_region)
    )

    @torch.autocast("cuda", dtype=torch.float32, enabled=False)
    def _vis_level(img: torch.Tensor, logits: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        logits = logits.float().sigmoid()

        # Sort gts, ensuring that the first masks have contents
        ordering = torch.argsort(gts.sum(dim=(-2, -1)), descending=True)
        gts = gts[ordering[:max_rows]]
        logits = logits[ordering[:max_rows]]

        # Create a grid of masks
        grid_loc = rearrange([gts, logits], "type num h w -> (num h) (type w)")
        grid_loc = repeat(grid_loc, "h w -> c h w", c=3)

        # Add the image to the grid
        grid_img = repeat(img, "c h w -> c (num h) (type w)", type=2, num=gts.shape[0])
        grid_img *= 0.33

        # Add error to the grid foreground
        err_values = rearrange(gts + logits - 1.0, "num h w -> (num h) w")
        err_color = torch.stack(
            [
                err_values.abs() * (err_values <= 0).float(),
                1.0 - err_values.abs(),
                err_values.abs() * (err_values > 0).float(),
            ],
            dim=0,
        )

        grid_loc = torch.cat([grid_loc, err_color], dim=2)
        grid_img = torch.cat([grid_img, torch.zeros_like(err_color)], dim=2)

        # Combined grid
        grid = reduce(torch.stack([grid_img, grid_loc]), "n c h w -> c h w", "max")

        return grid

    # reshape image to match the level shape
    img = F.interpolate(img.unsqueeze(0), size=pr_center.shape[-2:])[0]

    grid_thing = _vis_level(img, pr_center, gt_center)
    grid_stuff = _vis_level(img, pr_region, gt_region)

    return grid_thing, grid_stuff


def visualize_masks_stuff(logits: torch.Tensor, gts: torch.Tensor, *, index_mask: torch.Tensor, max_rows=4):
    index_mask = index_mask.detach()
    logits = logits[index_mask, ...].detach().sigmoid()
    gts = gts[index_mask, ...].detach()

    ordering = torch.argsort(gts.sum(dim=(-2, -1)), descending=True)
    gts = gts[ordering[:max_rows]]
    logits = logits[ordering[:max_rows]]

    # Create a grid of masks
    grid = rearrange([gts, logits], "type num h w -> () (num h) (type w)")

    yield grid


def visualize_masks_thing(
    logits: torch.Tensor,
    gts: torch.Tensor,
    *,
    index_mask: torch.Tensor,
    weighted_num: int,
    weighted_values: torch.Tensor,
    instance_num: int,
    **kwargs,
):
    logits = logits.detach()
    gts = gts.detach()
    index_mask = index_mask.detach()

    n, _, h, w = gts.shape

    logits = logits.reshape(n, instance_num, weighted_num, h, w)
    logits = logits.reshape(-1, weighted_num, h, w)

    gts = gts.unsqueeze(2).expand(n, instance_num, weighted_num, h, w)
    gts = gts.reshape(-1, weighted_num, h, w)

    weighted_any = weighted_values.reshape(-1, weighted_num) > 0
    index_mask = index_mask.unsqueeze(1) * weighted_any

    for _, grid in visualize_masks_stuff(logits, gts, index_mask=index_mask, **kwargs):
        yield grid


@torch.no_grad()
def visualize_true_things(images: torch.torch.Tensor, multithings: list[supervision.Things], max_batch=2):
    images = F.interpolate(images, scale_factor=0.75).squeeze(1).cpu()

    num_cats = multithings[0].scoremap.shape[1]
    supergrid = []
    for b in range(max_batch):
        if images.shape[0] <= b:
            break
        img = images[b]

        grid_img = repeat(img.cpu(), "c h w -> c h (n w)", n=num_cats)

        grid_items = []
        for things_level in multithings:
            things_level = things_level[b].detach()
            if things_level.num_instances == 0:
                grid_items.append(torch.ones_like(grid_img))
                continue

            scoremap = F.interpolate(things_level.scoremap.unsqueeze(1), size=img.shape[-2:]).squeeze(1).cpu()

            insts = things_level.instances.valid
            masks = F.interpolate(insts.insts.unsqueeze(1), size=img.shape[-2:]).squeeze(1).cpu()

            grid_masks = torch.zeros_like(scoremap)
            for mask, cat in zip(masks, insts.categories.cpu()):
                grid_masks[cat, :, :] += mask
            grid_masks = grid_masks.float().clamp(min=0, max=1)

            grid_scores = rearrange(scoremap, "n h w -> () h (n w)")
            grid_masks = rearrange(grid_masks, "n h w -> () h (n w)")
            grid_level = torch.cat([grid_masks, grid_scores, torch.zeros_like(grid_masks)], dim=0)  # RGB

            grid_items.append(grid_level)

        grid = torch.cat(grid_items, dim=1)

        grid_background = repeat(grid_img, "c h w -> c (n h) w", n=len(multithings))
        grid = reduce(torch.stack([grid_background * 0.3, grid]), "n c h w -> c h w", "max")
        grid = torch.cat([grid_img, grid], dim=1)

        supergrid.append(grid)
    return torch.cat(supergrid, dim=1)


@torch.no_grad()
def visualize_true_stuff(images: torch.torch.Tensor, multistuff: list[supervision.Stuff], max_batch=2):
    images = F.interpolate(images, scale_factor=0.75).squeeze(1).cpu()

    num_cats = multistuff[0].scoremap.shape[1]
    supergrid = []
    for b in range(max_batch):
        if images.shape[0] <= b:
            break
        img = images[b]

        grid_img = repeat(img.cpu(), "c h w -> c h (n w)", n=num_cats)

        grid_items = []
        for stuffs_level in multistuff:
            stuffs_level = stuffs_level[b].detach()
            if stuffs_level.num_instances == 0:
                grid_items.append(torch.zeros_like(grid_img))
                continue

            # Scoremap
            scoremap = F.interpolate(stuffs_level.scoremap.unsqueeze(1), size=img.shape[-2:]).squeeze(1).cpu()
            grid_scores = rearrange(scoremap, "n h w -> () h (n w)")

            # Instances
            select = stuffs_level.indices > 0

            masks = F.interpolate(stuffs_level.masks[select].float().unsqueeze(1), size=img.shape[-2:]).squeeze(1).cpu()
            labels = F.interpolate(stuffs_level.labels[select].unsqueeze(1), size=img.shape[-2:]).squeeze(1).cpu()

            grid_masks = torch.zeros_like(scoremap)
            grid_labels = torch.zeros_like(scoremap)
            for cat, (mask, label) in enumerate(zip(masks, labels)):
                grid_masks[cat, :, :] += mask
                grid_labels[cat, :, :] += label
            grid_masks = rearrange(grid_masks, "n h w -> () h (n w)")
            grid_labels = rearrange(grid_labels, "n h w -> () h (n w)")

            grid_level = torch.cat([grid_masks, grid_scores, grid_labels], dim=0)  # RGB
            grid_items.append(grid_level)

        grid = torch.cat(grid_items, dim=1)

        grid_background = repeat(grid_img, "c h w -> c (n h) w", n=len(multistuff))
        grid = reduce(torch.stack([grid_background * 0.3, grid]), "n c h w -> c h w", "max")
        grid = torch.cat([grid_img, grid], dim=1)

        supergrid.append(grid)

    return torch.cat(supergrid, dim=1)
