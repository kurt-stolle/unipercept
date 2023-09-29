import torch
from torch import Tensor


def visualize_kernels_similarity(preds: Tensor):
    preds = preds[0].detach()

    a_norm = preds / torch.linalg.vector_norm(preds, dim=0, keepdim=True).clamp(min=1e-7)
    yield "kernels", torch.mm(a_norm, a_norm.T).unsqueeze(0)
