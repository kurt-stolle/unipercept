from torchvision.tv_tensors import Mask as _Mask

from .registry import pixel_maps

__all__ = ["Mask"]

# NOTE: This is a hack to make `torchvision.tv_tensors.Mask` a registered pixel map datapoint.
pixel_maps.register(_Mask)


@pixel_maps.register
class Mask(_Mask):
    pass
