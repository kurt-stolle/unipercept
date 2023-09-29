import torch
import torch.nn as nn
import unipercept.data.points as _P
from typing_extensions import override

__all__ = ["DepthGuidedPanopticLoss", "PanopticGuidedTripletLoss"]


class PanopticGuidedTripletLoss(nn.Module):
    """
    Impements a depth-guided panoptic loss, as described in MonoDVPS

    L = max(d_p + d_0 - d_n, 0)
    """

    @override
    def forward(self, depth_features: torch.Tensor, segmentations: _P.PanopticMap):
        # TODO: merge
        return depth_features.mean() * 0.0


class DepthGuidedPanopticLoss(nn.Module):
    """
    Impements a depth-guided panoptic loss, as described in MultiDVPS
    """

    @override
    def forward(self, panseg_features: torch.Tensor, depths: _P.DepthMap):
        # TODO: merge
        return panseg_features.mean() * 0.0
