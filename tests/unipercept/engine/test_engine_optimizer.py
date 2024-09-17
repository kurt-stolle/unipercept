from __future__ import annotations

import pytest
import torch.optim
from unipercept.engine._optimizer import OptimizerFactory, OptimPackage, OptimType


@pytest.mark.parametrize("optim_package", [t for t in OptimPackage])
@pytest.mark.parametrize("optim_type", [t for t in OptimType])
@pytest.mark.parametrize("lr", [0.1])
@pytest.mark.parametrize("foreach", [True, False, None])
def test_optimizer_factory(model, optim_package, optim_type, lr, foreach):
    optim_args = {
        "lr": lr,
        "foreach": foreach,
    }
    try:
        optim_lazy = OptimizerFactory(optim_type, optim_package, **optim_args)
        optim = optim_lazy(model)
    except ImportError:
        pytest.skip("Optimizer not installed")
    except NotImplementedError:
        pytest.skip("Optimizer not implemented")
    except TypeError as t:
        err = str(t)
        if "unexpected keyword argument 'foreach'" in err:
            pytest.xfail(str(t))
        else:
            raise t
    assert isinstance(optim, torch.optim.Optimizer)


def test_optimizer_wraped(model):
    cls = torch.optim.SGD
    lazy = OptimizerFactory(cls, lr=0.1)
    optim = lazy(model)

    assert issubclass(type(optim), torch.optim.Optimizer) or isinstance(
        optim, torch.optim.Optimizer
    ), f"Expected {torch.optim.Optimizer}, got {type(optim)}"
