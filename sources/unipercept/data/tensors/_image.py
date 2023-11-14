from torchvision.tv_tensors import Image as _Image

from .registry import pixel_maps

__all__ = ["Image"]

# NOTE: This is a hack to make `torchvision.tv_tensors.Image` a registered pixel map datapoint.
pixel_maps.register(_Image)


@pixel_maps.register
class Image(_Image):
    pass
