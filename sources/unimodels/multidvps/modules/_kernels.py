from __future__ import annotations

from typing import Optional, Sequence, TypeAlias

from tensordict import TensorDict
from torch import Tensor, nn
from typing_extensions import override

__all__ = ["KernelSpaces", "Kernelizer", "GeometryEncoder"]

KernelSpaces: TypeAlias = TensorDict


class Kernelizer(nn.Module):
    """The kernel generator generates a space of kernel vectors (B, C, H, W) from a feature map (B, C, H, W)."""

    def __init__(self, heads: dict[str, nn.Module], encoder: nn.Module | None = None):
        super().__init__()

        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)

    def keys(self) -> Sequence[str]:
        return tuple(self.heads.keys())

    @override
    def forward(self, fe: Tensor) -> dict[str, Tensor]:
        """Apply single-level kernel head with optional common encoder."""

        if self.encoder is not None:
            fe = self.encoder(fe)

        k_spaces = {key: head(fe) for key, head in self.heads.items()}
        return TensorDict.from_dict(k_spaces, batch_size=[fe.shape[0]], device=fe.device)


class GeometryEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.mapper = nn.Conv2d(self.encoder.out_channels, out_channels, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.mapper(self.encoder(x)).sigmoid()
