import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import Tensor


def visualize_locations(
    img: Tensor, *, pr_center: Tensor, pr_region: Tensor, gt_center: Tensor, gt_region: Tensor, max_rows=2
) -> tuple[Tensor, Tensor]:
    """
    Yields tensor visualization for things and stuff as a tuple
    """

    img, pr_center, pr_region, gt_center, gt_region = map(
        lambda t: t.detach().cpu(), (img, pr_center, pr_region, gt_center, gt_region)
    )

    @torch.autocast("cuda", dtype=torch.float32, enabled=False)
    def _vis_level(img: Tensor, logits: Tensor, gts: Tensor) -> Tensor:
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


def visualize_masks_stuff(logits: Tensor, gts: Tensor, *, index_mask: Tensor, max_rows=4):
    index_mask = index_mask.detach()
    logits = logits[index_mask, ...].detach().sigmoid()
    gts = gts[index_mask, ...].detach()

    ordering = torch.argsort(gts.sum(dim=(-2, -1)), descending=True)
    gts = gts[ordering[:max_rows]]
    logits = logits[ordering[:max_rows]]

    # Create a grid of masks
    grid = rearrange([gts, logits], "type num h w -> () (num h) (type w)")

    yield "stuff_masks", grid


def visualize_masks_thing(
    logits: Tensor,
    gts: Tensor,
    *,
    index_mask: Tensor,
    weighted_num: int,
    weighted_values: Tensor,
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
        yield "thing_masks", grid
