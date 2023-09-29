import torch
from torch import Tensor


def visualize_depths(preds: Tensor, gts: Tensor):
    pr = preds[0].detach()
    gt = gts[0].detach()

    yield "depths", torch.zeros(pr.shape[-2:], dtype=torch.float32).cpu()
