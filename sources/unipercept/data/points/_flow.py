from torchvision.datapoints import Mask

from .registry import pixel_maps

__all__ = ["OpticalFlow"]


@pixel_maps.register
class OpticalFlow(Mask):
    pass
