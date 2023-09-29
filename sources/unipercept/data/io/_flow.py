from __future__ import annotations

import torch
from unicore import file_io
from unipercept.data.points import OpticalFlow

__all__ = ["read_optical_flow"]


@torch.inference_mode()
@file_io.with_local_path(force=True)
def read_optical_flow(path: str) -> torch.Tensor:
    from flowops import Flow

    flow = torch.from_numpy(Flow.read(path).as_2hw())
    return OpticalFlow(flow.to(dtype=torch.float32))
