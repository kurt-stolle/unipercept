from typing import Callable, Iterable, Optional

import torch

__all__ = ["cat_nonempty", "gather_feature", "topk_score"]


def cat_nonempty(tensors: Iterable[torch.Tensor | None], dim=0) -> Optional[torch.Tensor]:
    """
    Concatenate an interable of tensors for each entry that has a length greater
    than zero.

    Parameters
    ----------
    tensors
        Iterable of `Tensor` objects.
    dim, optional
        Dimension to concatenate, by default 0

    Returns
    -------
        Concatenated tensor.
    """
    tensors = list(tensors)
    tensors_nonempty = [t for t in tensors if t is not None]
    if len(tensors_nonempty) == 0:
        return None
    return torch.cat(tensors_nonempty, dim=dim)


def topk_score(scores, *, K: int, score_shape: torch.Size):
    """
    Get the top $k$ points in a scope mat.

    Parameters
    ----------
    scores
        Map of scores.
    K
        Amount of points to select.
    score_shape
        Original shape of the score map.

    Returns
    -------
        Tensors of the Top-K (scores, indices, classes, y, x)
    """
    batch, channel, height, width = score_shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor").float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = torch.div(index, K, rounding_mode="floor")
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feature(
    fmap: torch.Tensor, index: torch.Tensor, mask: Optional[torch.Tensor] = None, use_transform: bool = False
) -> torch.Tensor:
    """
    Gather features from a mapping.

    Parameters
    ----------
    fmap
        Feature map.
    index
        Index tensor.
    mask, optional
        Masking tensor, by default None
    use_transform, optional
        Flag to apply transformation, by default False

    Returns
    -------
        Tensor with gathered features.
    """
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap
