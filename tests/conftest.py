"""
Global PyTest configuration
"""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTEST_ADDOPTS"] = "--color=yes"

import pytest
import torch
import torch._dynamo

print(f"PyTorch version: {torch.__version__}")

torch.autograd.set_detect_anomaly(True)


@pytest.fixture(params=["cpu", "cuda"], ids=lambda p: f"{p.upper()}")
def device(request):
    import torch

    device_name = request.param
    if device_name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available!")
        else:
            # pytest.skip("PyTest limited")
            # The test should have the 'gpu' mark to skip if CUDA is not available

            pytest.mark.gpu(request.node)
            yield torch.device(device_name)
            torch.cuda.empty_cache()
    else:
        yield torch.device(device_name)
