from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from unipercept.nn import backbones

LAYER_SCOPE = "function"
BACKBONES = [
    *[
        ("timm", name) for name in ("convnextv2_tiny", "resnet50")
    ],  # "swin_tiny_patch4_window7_224"
    *[("torchvision", name) for name in ("swin_v2_s", "resnet50")],  # "convnext_tiny"
]


@pytest.fixture(
    params=BACKBONES,
    scope=LAYER_SCOPE,
    ids=[f"{src.title()}:{name.title()}" for src, name in BACKBONES],
)
def backbone(request):
    bb_src, bb_name = request.param

    match bb_src:
        case "timm":
            avail = backbones.timm.list_available(bb_name)
            assert (
                bb_name in avail
            ), f"{bb_name} not available, did you mean: {avail[0]}?"
            bb = backbones.timm.TimmBackbone(bb_name, pretrained=True)
        case "torchvision":
            avail = backbones.torchvision.list_available(bb_name)
            assert (
                bb_name in avail
            ), f"{bb_name} not available, did you mean: {avail[0]}?"
            bb = backbones.torchvision.TorchvisionBackbone(bb_name, weights=None)
        case _:
            raise ValueError(f"Invalid backbone source: {bb_src}")

    return bb


def test_backbones(device, backbone):
    backbone = backbone.to(device)
    x = torch.randint(0, 255, (1, 3, 256, 512), dtype=torch.uint8, device=device)
    fs = backbone(x)
    assert isinstance(x, torch.Tensor)
    assert isinstance(fs, TensorDict)

    assert len(fs.keys()) == len(backbone.feature_info)


# @pytest.fixture(
#     scope=LAYER_SCOPE,
#     params=[
#         backbones.fpn.WeightMethod.ATTENTION,
#         backbones.fpn.WeightMethod.SUM,
#         backbones.fpn.WeightMethod.FAST_ATTENTION,
#     ],
# )
# def fpn_weight_method(request):
#     return request.param


# @pytest.fixture(scope=LAYER_SCOPE, params=[0, 1, 2])
# def fpn_num_hidden(request):
#     return request.param


# @pytest.fixture(
#     scope=LAYER_SCOPE,
#     params=[backbones.fpn.build_default_routing, backbones.fpn.build_pan_routing, backbones.fpn.build_quad_routing],
#     ids=["D", "P", "Q"],
# )
# def fpn_routing(request, fpn_weight_method):
#     return functools.partial(request.param, weight_method=fpn_weight_method)


# @pytest.fixture(
#     scope=LAYER_SCOPE,
# )
# def fpn(backbone, fpn_num_hidden, fpn_routing):
#     routing = fpn_routing(num_levels=6)

#     return backbones.fpn.FeaturePyramidBackbone(
#         backbone,
#         out_channels=16,
#         num_hidden=fpn_num_hidden,
#         routing=routing,
#         in_features=list(backbone.feature_info.keys()),
#     )


# def test_fpn_valid_input(device, fpn):
#     fpn = fpn.to(device)
#     x = torch.randn(1, 3, 256, 512, requires_grad=True, device=device)
#     ys = fpn(x)
#     assert isinstance(ys, TensorDict)

#     loss = torch.stack([y.sum() for y in ys.values()]).sum()
#     loss.backward()

#     g = x.grad
#     assert g is not None

#     print([y.shape for y in ys])
#     print(f"Gradient: {g.shape} (mean = {g.mean().item()}, std = {g.std().item()})")
