r"""
Wrapper for loading datasets from HuggingFace.

See Also
--------
- `HuggingFace Datasets <https://huggingface.co/docs/datasets/>`_
- `Relevant documentation <https://huggingface.co/docs/datasets/use_with_pytorch>`_
"""

from __future__ import annotations

import dataclasses as D
import typing as T

import typing_extensions as TX

try:
    import datasets as hfds
    import huggingface_hub as hfhub
except ImportError:
    raise ImportError(
        "Huggingface datasets not installed. Install with `pip install datasets[vision] huggingface_hub`",
    )

from . import (
    Manifest,
    PerceptionDataset,
)

__all__ = []


class HuggingfaceDataset(PerceptionDataset, id="huggingface"):
    path: str
    split: str
    config: dict[str, T.Any] = D.field(
        default_factory=dict,
        repr=False,
        metadata={
            "help": (
                "Keyword arguments passed to ``load_dataset``. See the HuggingFace documentation for more information."
            )
        },
    )

    def _build_source(self) -> hfds.Dataset:
        return hfds.load_dataset(self.path, split=self.split, **self.config)

    @TX.override
    def _build_manifest(self) -> Manifest:
        # TODO
        raise NotImplementedError()


if __name__ == "__main__":
    ds = HuggingfaceDataset(path="scene_parse_150", split="train")
