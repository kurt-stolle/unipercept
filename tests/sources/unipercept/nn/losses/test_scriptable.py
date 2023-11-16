import torch
import pytest
from functools import partial
import unipercept.nn.losses as _L

LOSS_NN_MODULES = [
    _L.DepthLoss,
    _L.WeightedThingDiceLoss,
    _L.WeightedStuffDiceLoss,
    _L.PGTLoss,
    _L.DGPLoss,
    _L.PEDLoss,
    partial(_L.SigmoidFocalLoss, alpha=0.33, gamma=1.8),
]


@pytest.fixture(
    scope="module", params=LOSS_NN_MODULES, ids=[m.__name__ if isinstance(m, type) else str(m) for m in LOSS_NN_MODULES]
)
def loss_module(request):
    mod = request.param()
    return mod


def test_loss_scriptable(loss_module):
    print(f"Module: {loss_module}")
    torch.jit.script(loss_module)
    print(f"Module successfully scripted!")
