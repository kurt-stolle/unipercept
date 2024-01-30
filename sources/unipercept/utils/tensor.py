from __future__ import annotations

import typing as T

import torch

__all__ = []


def cat_nonempty(
    tensors: T.Iterable[torch.Tensor | None], dim=0
) -> T.Optional[torch.Tensor]:
    """
    Concatenate an interable of tensors for each entry that has a length greater
    than zero.

    Parameters
    ----------
    tensors
        T.Iterable of `Tensor` objects.
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
    fmap: torch.Tensor,
    index: torch.Tensor,
    mask: T.Optional[torch.Tensor] = None,
    use_transform: bool = False,
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


def map_values(
    x: torch.Tensor,
    translation: torch.Tensor
    | T.Tuple[torch.Tensor, torch.Tensor]
    | T.Dict[int, int | float | bool],
    default: int | float | bool | None = None,
) -> torch.Tensor:
    """
    Maps the values in a tensor to a new set of values.

    Parameters
    ----------
    x
        Input tensor.
    translation
        Mapping from old values to new values.

    Returns
    -------
    torch.Tensor
        Tensor with mapped values.


    Examples
    --------
    >>> t = (torch.arange(0, 256), torch.randperm(256))
    >>> x = torch.randint(0, 256, (16, 224, 224, 3))
    >>> y = map_values(x, t)
    """

    # Find the translation sources and target values
    if isinstance(translation, dict):
        assert all(
            isinstance(k, int) for k in translation.keys()
        ), "Expected integer keys"
        t_from = torch.tensor(translation.keys(), dtype=torch.int64, device=x.device)
        t_to = torch.tensor(translation.values(), device=x.device)
    elif isinstance(translation, torch.Tensor):
        assert (
            translation.ndim == 2
        ), f"Expected 2D tensor, got {translation.ndim}D tensor"
        t_from = translation[0, :]
        t_to = translation[1, :]
    else:
        assert (
            isinstance(translation, tuple) and len(translation) == 2
        ), "Expected tuple of length 2"
        t_from, t_to = translation
        assert t_from.ndim == 1 and t_to.ndim == 1, "Expected 1D tensors"

    # Ensure that values in `x` are in `t_from`
    if default is not None:
        v_unique = torch.unique(x)
        t_missing = ~torch.isin(v_unique, t_from)

        # Add missing values
        t_from = torch.cat([t_from, v_unique[t_missing]])
        t_to = torch.cat([t_to, torch.full_like(v_unique[t_missing], default)])

        # Sort the values (for bucketize)
        t_from_sort_indices = torch.argsort(t_from)
        t_from = t_from[t_from_sort_indices]
        t_to = t_to[t_from_sort_indices]

    assert t_from.shape == t_to.shape, "Expected tensors of the same shape"
    assert torch.all(
        torch.isin(x, t_from)
    ), f"Not all values in `x` ({x.detach().unique().cpu().tolist()}) are in `t_from` ({t_from.detach().cpu().tolist()}))"

    # Map the values
    i = torch.bucketize(x.ravel(), t_from)
    y = t_to[i]

    return y.reshape(x.shape)
