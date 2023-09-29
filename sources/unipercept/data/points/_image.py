from torchvision.datapoints import Image as _Image

from .registry import pixel_maps

__all__ = ["Image"]

# NOTE: This is a hack to make `torchvision.datapoints.Image` a registered pixel map datapoint.
pixel_maps.register(_Image)


@pixel_maps.register
class Image(_Image):
    pass
