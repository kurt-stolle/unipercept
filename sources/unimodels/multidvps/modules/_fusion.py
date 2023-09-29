"""Implements kernel fusion and mapping to task-specific kernels."""

from __future__ import annotations

import typing as T

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from typing_extensions import override

__all__ = ["KernelFusion", "ThingFusion", "StuffFusion"]


class KernelFusion(nn.Module):
    """Baseclass for kernel fusion modules."""

    def __init__(self, key: str, dims: int, mapping: dict[str, nn.Module]):
        super().__init__()

        linear = nn.Linear(dims, dims, bias=True)
        nn.init.orthogonal_(linear.weight)
        nn.utils.parametrizations.orthogonal(linear, name="weight")

        self.key = key
        self.input = TensorDictModule(
            linear,
            in_keys=[key],
            out_keys=[key],
        )
        self.mapper = TensorDictSequential(
            *[TensorDictModule(module, in_keys=[key], out_keys=[to]) for to, module in mapping.items()]
        )

    @override
    def forward(
        self, kernels: TensorDict, categories: Tensor | None, scores: Tensor | None
    ) -> tuple[TensorDict, Tensor | None, Tensor | None]:
        kernels = self.input(kernels)
        kernels = self.mapper(kernels)  # todo: before or after?

        if not self.training:
            assert categories is not None
            assert scores is not None

            kernels, categories, scores = self.kernel_fusion(kernels, categories, scores)

        return kernels, categories, scores

    if T.TYPE_CHECKING:
        __call__ = forward

    def kernel_fusion(
        self, kernels: TensorDict, categories: Tensor, scores: Tensor
    ) -> tuple[TensorDict, Tensor, Tensor]:
        """Fuse all kernels that belong to the same category."""
        batch = kernels.batch_size[0]
        assert batch == 1, f"Batch size must be 1, got {batch}"

        k: TensorDict = kernels.squeeze(0)  # type: ignore
        k, c, s = self._kernel_fusion(k, categories.squeeze(0), scores.squeeze(0))

        kernels = k.view(batch, *k.shape)  # type: ignore
        return kernels, c.view(batch, *c.shape), s.view(batch, *s.shape)

    def _kernel_fusion(
        self, kernels: TensorDict, categories: Tensor, scores: Tensor
    ) -> tuple[TensorDict, Tensor, Tensor]:
        raise NotImplementedError()


class StuffFusion(KernelFusion):
    """Fuse kernels that belong to the same category."""

    @override
    def _kernel_fusion(self, kernels: TensorDict, categories: Tensor, scores: Tensor):
        """Fuse all kernels that belong to the same category for stuff."""
        uniq = torch.unique(categories)

        mat_cats = categories.unsqueeze(0)
        mat_uniq = uniq.unsqueeze(1)

        label_matrix = (mat_cats == mat_uniq).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)

        scores = torch.mm(label_matrix, scores.unsqueeze(-1)) / label_norm
        scores.squeeze_(-1)

        def fuse(k: Tensor) -> Tensor:
            k = torch.mm(label_matrix, k)
            k /= label_norm

            return k

        kernels = kernels.apply(fuse, batch_size=scores.shape)  # type: ignore

        return kernels, uniq, scores


class ThingFusion(KernelFusion):
    """Fuses kernels that are similar to each other."""

    def __init__(self, *args, fusion_threshold: float, mask_categories=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.fusion_threshold = fusion_threshold
        self.mask_categories = mask_categories
        self.similarity = CosineSelfSimilarity()

    @override
    def _kernel_fusion(self, kernels: TensorDict, categories: Tensor, scores: Tensor):
        """
        Fuse all kernels that have a high similarity for things.
        Kernel weights are considered identical if the cosine similarity surpasses threshold ``self.sim_thres``.
        """

        sim_emb = kernels.get(self.key)
        sim_matrix = self.similarity(sim_emb, 1)

        # The method `Tensor.trui` returns the upper triagonal part
        # of a matrix or batch of matrices by setting other elements to zero.
        label_matrix = sim_matrix.triu() >= self.fusion_threshold

        # If class-specific, apply a mask to the label matrix
        # The categories that are not the same are set to zero.
        if self.mask_categories:
            cate_matrix = categories.unsqueeze(-1) == categories
            label_matrix = label_matrix & cate_matrix

        cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2

        # Get a view of the diagonal to find keep indices
        keep_matrix = cum_matrix.diagonal()

        label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)

        categories = categories[keep_matrix]
        scores = scores[keep_matrix]

        def fuse(k: Tensor) -> Tensor:
            k = torch.mm(label_matrix, k)
            k /= label_norm

            return k

        kernels = kernels.apply(fuse, batch_size=scores.shape)  # type: ignore

        # Apply label matrix to the kernel weights
        return kernels, categories, scores


class CosineSelfSimilarity(nn.Module):
    """Measures the cosine similarity between all pairs of kernels."""

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    @override
    def forward(self, a: Tensor, dim: int) -> Tensor:
        """
        Manual computation of the cosine similarity.

        Based on: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re  # noqa: E501

        ```
        csim(a,b) = dot(a, b) / (norm(a) * norm(b))
                = dot(a / norm(a), b / norm(b))
        ```

        Clamp with min(eps) for numerical stability. The final dot
        product is computed via transposed matrix multiplication (see `torch.mm`).
        """
        a_norm = self._stable_norm(a, dim)
        return torch.mm(a_norm, a_norm.T)

    def _stable_norm(self, t: torch.Tensor, dim: int) -> Tensor:
        return t / torch.linalg.vector_norm(t, dtype=torch.float, dim=dim, keepdim=True).clamp(self.eps)
