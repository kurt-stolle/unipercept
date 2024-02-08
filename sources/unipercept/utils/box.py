from __future__ import annotations

import torch


@torch.jit.script_if_tracing
def boxes_to_areas(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(boxes.shape[:-1], device=boxes.device, dtype=boxes.dtype)

    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]

    return w * h
