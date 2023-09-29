from __future__ import annotations

import typing as T

import torch
from detectron2.utils.events import get_event_storage

if T.TYPE_CHECKING:
    import unimodels.multidvps.modules as _M
    import unimodels.multidvps.visuals as _V

__all__ = ["is_visualization_iteration", "visualize_locations", "visualize"]


def is_visualization_iteration(vis_period: int):
    try:
        storage = get_event_storage()
    except AssertionError:
        storage = None
    return storage is not None and vis_period > 0 and storage.iter % vis_period == 0


def visualize_locations(images: torch.Tensor, multiloc: _M.MultiLevelDetections, gts: _M.supervision.Truths):
    return  # FIXME
    img = images[0].detach().cpu()

    for level, (loc, gt_thing, gt_stuff) in enumerate(zip(multiloc, gts.thing, gts.stuff)):
        pr_region = loc.stuff_map[0]
        pr_center = loc.thing_map[0]

        gt_center = gt_thing.scoremap[0]
        gt_region = gt_stuff.scoremap[0]

        grid_thing, grid_stuff = _V.visualize_locations(
            img, pr_center=pr_center, pr_region=pr_region, gt_center=gt_center, gt_region=gt_region
        )

        yield f"thing_centers/level_{level}", grid_thing
        yield f"stuff_regions/level_{level}", grid_stuff


_MultiParams = T.ParamSpec("_MultiParams")
_R = T.TypeVar("_R")


def visualize(
    fn: T.Callable[_MultiParams, T.Iterator[tuple[str, torch.Tensor]]], *format_args, **format_kwargs
) -> T.Callable[_MultiParams, None]:
    """
    Wrapper for a method that visualizes the output of a function. The wrapper ensures that the inputs are
    detached and the outputs are put into the event storage.
    """

    @torch.autocast("cuda", enabled=False)
    def wrapper(*args: _MultiParams.args, **kwargs: _MultiParams.kwargs) -> None:
        # Detach
        args = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args)  # type: ignore
        kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}  # type: ignore

        with torch.no_grad():
            for name, tensor in fn(*args, **kwargs):
                tensor = tensor.cpu()
                name = name.format(*format_args, **format_kwargs)
                get_event_storage().put_image(name, tensor)
                del tensor

    return wrapper
