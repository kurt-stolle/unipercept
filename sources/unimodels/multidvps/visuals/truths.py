import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from unipercept.modeling import supervision


def visualize_true_things(images: torch.Tensor, multithings: list[supervision.Things], max_batch=2):
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
                grid_items.append(torch.zeros_like(grid_img))
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

    supermap = torch.cat(supergrid, dim=1)

    if len(supergrid) > 0:
        yield "supervision/thing/scoremap", supermap


def visualize_true_stuff(images: torch.Tensor, multistuff: list[supervision.Stuff], max_batch=2):
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

    supermap = torch.cat(supergrid, dim=1)

    if len(supergrid) > 0:
        yield "supervision/stuff/scoremap", supermap
