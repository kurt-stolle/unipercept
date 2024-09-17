"""
Provides a reference implementation in PyTorch for when the extension is not available.
"""

from __future__ import annotations

from torch import stack
from torch.nn.functional import grid_sample

__all__ = []  # no exports to encourage explicit import of the CPU implementation


def deform_attn(values: Tensor, shapes: Tensor, locs: Tensor, attn: Tensor):
    """
    Implements the forward pass.

    See Also
    --------

    - ``extension.cpp`` : corresponding implementation.
    """
    N, _, M, D = values.shape  # [N, S, M, D]
    assert locs.shape[2] == M, (values.shape, locs.shape)

    _, Q, M, L, P, _ = locs.shape  # [N, Q, M, L, P, 2]

    value_list = values.split([H * W for H, W in shapes], dim=1)

    sampling_grids = 2 * locs - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attn = attn.transpose(1, 2).reshape(N * M, 1, Q, L * P)
    output = (
        (stack(sampling_value_list, dim=-2).flatten(-2) * attn)
        .sum(-1)
        .view(N, M * D, Q)
    )
    return output.transpose(1, 2).contiguous()
