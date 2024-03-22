from __future__ import annotations

import enum
import functools
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    Literal,
    Mapping,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import torch

from unipercept.file_io import get_local_path
from unipercept.utils.typings import Pathable

__all__ = ["multi_read", "NoEntriesAction", "get_kwd", "read_pixels"]


def read_pixels(path: str, color: bool, alpha=False) -> torch.Tensor:
    """Read an image file using OpenCV."""
    import cv2
    import numpy as np

    if alpha:
        flags = cv2.IMREAD_UNCHANGED
    else:
        flags = (
            cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        ) | cv2.IMREAD_ANYDEPTH

    image = cv2.imread(path, flags)
    if image is None:
        raise RuntimeError(f"Failed to read image at path: {path}")

    if color and alpha:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor_image = torch.from_numpy(image.astype(np.int32))

    return tensor_image


def write_png(path: Pathable, tensor: torch.Tensor):
    """Write a tensor to a PNG file."""
    import cv2

    mat = tensor.cpu().numpy()
    cv2.imwrite(get_local_path(path), mat)


_KeywordType = TypeVar("_KeywordType")


def get_kwd(
    kwds_dict: dict[str, Any], name: str, cast_as: _KeywordType, /
) -> _KeywordType:
    try:
        res = kwds_dict.pop(name)
    except KeyError as e:
        raise TypeError(f"Missing keyword argument: {e.args[0]!r}") from e

    return cast(_KeywordType, res)


_ReadParams = ParamSpec("_ReadParams")
_ReadReturn = TypeVar("_ReadReturn", bound=torch.Tensor)


class NoEntriesAction(enum.StrEnum):
    """Enum to specify the action to take when no entries are found in a source."""

    ERROR = enum.auto()
    NONE = enum.auto()


@overload
def multi_read(
    reader: Callable[Concatenate[str, _ReadParams], _ReadReturn],
    key: Any,
    *,
    no_entries: Literal[NoEntriesAction.ERROR] | Literal["error"],
) -> Callable[Concatenate[Sequence[Mapping[Any, Any]], _ReadParams], _ReadReturn]:
    ...


@overload
def multi_read(
    reader: Callable[Concatenate[str, _ReadParams], _ReadReturn],
    key: Any,
    *,
    no_entries: Literal[NoEntriesAction.NONE] | Literal["none"] = NoEntriesAction.NONE,
) -> Callable[
    Concatenate[Sequence[Mapping[Any, Any]], _ReadParams], _ReadReturn | None
]:
    ...


def multi_read(
    reader: Callable[Concatenate[str, _ReadParams], _ReadReturn],
    key: Any,
    *,
    no_entries: NoEntriesAction | str = NoEntriesAction.NONE,
) -> Callable[
    Concatenate[Sequence[Mapping[Any, Any]], _ReadParams], _ReadReturn | None
]:
    """
    Call a reader function multiple times and stack the results into a single tensor of the
    same type as the first result.

    Accepts callables where the first argument is a string, mapped to `path` in the sources, and the
    remaining arguments are keywords mapped to the same key in the meta-entry (if it exists).

    The returned callable accepts the same arguments as the reader function, but the first argument
    is a list of sources. The arguments in the "meta" entry of the source are

    Example:
    ```python
    from unipercept.data.io import multi_read, read_panoptic_map

    sources = [
        {
            "panoptic": {
                "path": "panoptic1.png",
                "meta": {
                    "format": "cityscapes",
                },
            },
            "random_key_1": "foo"
        },
        {
            "panoptic": {
                "path": "panoptic2.png",
                "meta": {
                    "format": "cityscapes",
                },
            },
            "random_key_2": "bar"
        },
    ]

    dataset_info = {
        "label_divisor": 1000,
        "ignore_label": 255,
        "stuff_translations": {
            10: 0,
            12: 1,
            23: 2,
        },
    }

    panoptic_maps = multi_read(read_panoptic_map, key="panoptic")(sources, dataset_info)
    ```

    The result is a single tensor with shape `(2, H, W)` of type ``PanopticMap``.
    """

    @functools.wraps(reader)
    def wrapped(
        sources: Sequence[Mapping[Any, Any]],
        *args: _ReadParams.args,
        **kwargs: _ReadParams.kwargs,
    ) -> _ReadReturn | None:
        if not isinstance(sources, Iterable):
            raise TypeError("The first argument must be an iterable of sources")

        if not all(isinstance(s, Mapping) for s in sources):
            raise TypeError("All sources must be mappings")

        if not any(key in s for s in sources):
            match NoEntriesAction(no_entries):
                case NoEntriesAction.ERROR:
                    raise ValueError(
                        f"At least one of sources must have the '{key}' key"
                    )
                case NoEntriesAction.NONE:
                    return None
                case _:
                    raise TypeError(f"Invalid value for 'no_entries': {no_entries!r}")

        if not all(isinstance(s[key], Mapping) for s in sources if key in s):
            raise TypeError(f"All sources must have a mapping at the '{key}' key")

        res_list: list[_ReadReturn | None] = [None] * len(sources)
        for i, s in enumerate(map(lambda s: s.get(key, None), sources)):
            if s is None:
                continue
            if not isinstance(s, Mapping):
                raise TypeError(f"The '{key}' key must be a mapping")
            item_path = s["path"]

            if "meta" in s:
                item_reader = functools.partial(reader, item_path, **s["meta"])
            else:
                item_reader = functools.partial(reader, item_path)
            item_obj = item_reader(*args, **kwargs)

            res_list[i] = item_obj

        _apply_imputation_with_default(res_list)

        if not all(isinstance(r, type(res_list[0])) for r in res_list):
            raise TypeError(
                f"All results must be of the same type, got: {[r for r in res_list]!r}"
            )

        res_type = type(res_list[0])
        assert issubclass(res_type, torch.Tensor)

        res_tensor = torch.stack(res_list)  # type: ignore

        return res_tensor.as_subclass(res_type)

    return wrapped


def _apply_imputation_with_default(res_list: list[_ReadReturn | None]) -> None:
    missing = [i for i, x in enumerate(res_list) if x is None]
    if len(missing) == res_list:
        raise RuntimeError("All entries are missing")
    elif len(missing) == 0:
        # Nothing to do, all are present
        return

    # Fill missing entries with defaults/zeros
    not_missing = next(m for m in res_list if m is not None)

    if hasattr(not_missing, "default_like"):
        default_fn = not_missing.default_like  # type: ignore
    elif hasattr(not_missing, "zeros_like"):
        default_fn = not_missing.zeros_like  # type: ignore
    else:
        raise TypeError(f"Cannot impute missing entries of type {type(not_missing)}")

    assert callable(default_fn)

    for i in missing:
        res_list[i] = default_fn(not_missing)
