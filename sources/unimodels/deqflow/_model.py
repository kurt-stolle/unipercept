from __future__ import annotations

try:
    import deqflow2
except ImportError:
    raise ImportError(
        "DEQFlow is not installed. Please install it the fork at: https://github.com/kurt-stolle/deq-flow"
    )


import typing as T

import torch
import torch.nn as nn
from typing_extensions import override
from unipercept.data.points import InputData


class DEQFlowWrapper(nn.Module):
    """Wraps DEQFlow (v2) in a way that is compatible with the UniPercept API."""

    def __init__(self, deqflow: deqflow2.DEQFlow):
        super().__init__()
        self.deqflow = deqflow

    @override
    def forward(self, inputs: InputData) -> T.Dict[str, torch.Tensor]:
        assert not self.training, "DEQFlow training in the UniPercept API is untested, and disabled in code for now."

        image_pair = inputs.captures.images
        assert (
            image_pair.ndim == 5
        ), f"DEQFlow requires batched pairs of RGB images as input, i.e. BPCHW, got {image_pair.ndim} dimensions!"
        assert (
            image_pair.shape[1] == 2
        ), f"DEQFlow inputs (BPCHW) must have P dimension size 2, got {image_pair.shape[1]}!"

        image1 = image_pair[:, 0]
        image2 = image_pair[:, 1]

        return self.deqflow(
            image1,
            image2,
            flow_gt=None,
            valid=None,
            fc_loss=None,
            flow_init=None,
            cached_result=None,
            writer=None,
            sradius_mode=False,
        )
