"""Pseudo label generation manager for finding missing labels during dataset discovery."""

from __future__ import annotations

import functools
import typing as T

import numpy as np
import PIL.Image as pil_image
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from unipercept import file_io
from unipercept.data.pipes import PILImageLoaderDataset
from unipercept.log import logger
from unipercept.utils.memory import find_executable

from ..types import Pathable

if T.TYPE_CHECKING:
    from transformers.pipelines import (
        DepthEstimationPipeline,
        ImageSegmentationPipeline,
    )

from unipercept.data.tensors import DepthFormat, DepthMap, LabelsFormat, PanopticMap


class PseudoGenerator:
    """
    Implements a pseudo label generator that uses HuggingFace Transformers
    to generate pseudo labels for missing labels during dataset discovery.

    Currently limited to DPT and Mask2Former models, and the target tasks are hardcoded.

    Notes
    -----

    This class is a good candidate for refactoring into a more generic class.
    """

    def __init__(
        self,
        depth_model="facebook/dpt-dinov2-large-kitti",  # "sayakpaul/glpn-kitti-finetuned-diode-221214-123047",
        depth_format: DepthFormat | str | None = None,
        panoptic_model="facebook/mask2former-swin-large-cityscapes-panoptic",
        panoptic_format: LabelsFormat | str | None = None,
    ):
        self._depth_model: T.Final[str] = depth_model
        self._depth_generate_queue: list[tuple[Pathable, Pathable]] = []
        self._depth_format = depth_format

        self._panoptic_model: T.Final[str] = panoptic_model
        self._panoptic_generate_queue: list[tuple[Pathable, Pathable]] = []

        self._panoptic_merge_queue: list[tuple[Pathable, Pathable, Pathable]] = []
        self._panoptic_format = panoptic_format

    def __len__(self) -> int:
        return len(self._depth_generate_queue)

    def __enter__(self) -> T.Self:
        return self

    def __exit__(self, *args) -> None:
        self.submit()

    def submit(self):
        self.run_depth_generator_queue()
        self.run_panoptic_generate_queue()
        self.run_panoptic_merge_queue()

    # --------------------- #
    # Panoptic Segmentation #
    # --------------------- #

    @functools.cached_property
    def _panoptic_pipeline(self) -> ImageSegmentationPipeline:
        from transformers import pipeline

        logger.info("Loading panoptic segmentation pipeline: %s", self._depth_model)

        pl = pipeline(
            task="image-segmentation",
            model=self._panoptic_model,
            device="cuda",
        )

        return T.cast("ImageSegmentationPipeline", pl)

    def add_panoptic_generate_task(
        self, *, image_path: Pathable, panoptic_path: Pathable
    ) -> None:
        self._panoptic_generate_queue.append((image_path, panoptic_path))

    def run_panoptic_generate_queue(self):
        if len(self._panoptic_generate_queue) == 0:
            return
        msg = "Panoptic segmentation is not yet implemented!"
        raise NotImplementedError(msg)

    # ---------------- #
    # Panoptic Merging #
    # ---------------- #

    def add_panoptic_merge_task(
        self,
        semantic_path: Pathable,
        instance_path: Pathable,
        panoptic_path: Pathable,
    ) -> None:
        self._panoptic_merge_queue.append((semantic_path, instance_path, panoptic_path))

    def run_panoptic_merge_task(
        self,
        semantic: pil_image.Image,
        instance: pil_image.Image,
        panoptic_path: Pathable,
    ) -> PanopticMap:
        from unipercept import file_io
        from unipercept.data.tensors import PanopticMap

        seg = torch.from_numpy(np.asarray(semantic).astype(np.int32))
        ins = torch.from_numpy(np.asarray(instance).astype(np.int32))

        if seg.shape != ins.shape:
            msg = f"Expected same size, got {seg.shape} and {ins.shape}!"
            raise ValueError(msg)
        if not (seg.ndim == ins.ndim == 2):
            msg = f"Expected 2D, got {seg.ndim} and {ins.ndim}!"
            raise ValueError(msg)

        pan = PanopticMap.from_parts(semantic=seg.cpu(), instance=ins.cpu())

        panoptic_path = file_io.Path(panoptic_path)
        pan.save(panoptic_path, format=self._panoptic_format)

        assert panoptic_path.exists(), f"Panoptic map not saved to {panoptic_path}!"

        return pan

    def run_panoptic_merge_queue(self):
        from unipercept import file_io

        if len(self._panoptic_merge_queue) == 0:
            return

        for items in tqdm(self._panoptic_merge_queue, desc="Panoptic merging"):
            semantic_path, instance_path, panoptic_path = map(file_io.Path, items)
            semantic = pil_image.open(semantic_path)
            instance = pil_image.open(instance_path)
            try:
                self.run_panoptic_merge_task(semantic, instance, panoptic_path)
            except Exception as e:
                msg = f"Failed to merge panoptic maps: ({semantic_path}, {instance_path}) -> {panoptic_path}"
                raise ValueError(msg) from e

    # ---------------- #
    # Depth Estimation #
    # ---------------- #

    @functools.cached_property
    def _depth_pipeline(self) -> DepthEstimationPipeline:
        from transformers import pipeline

        logger.info("Loading depth estimation pipeline: %s", self._depth_model)

        pl = pipeline(
            task="depth-estimation",
            model=self._depth_model,
            device="cuda",
        )

        return T.cast("DepthEstimationPipeline", pl)

    def add_depth_generator_task(
        self,
        image_path: Pathable,
        depth_path: Pathable,
    ) -> None:
        """Add a depth generation task to the queue."""
        self._depth_generate_queue.append((image_path, depth_path))

    def run_depth_generator_queue(self):
        """Run the depth generator on the queue."""
        from torchvision.transforms.v2.functional import (
            resize_image,
            to_pil_image,
            to_tensor,
        )

        if len(self._depth_generate_queue) == 0:
            return

        # Use a dataset to load the images in parallel
        ds = PILImageLoaderDataset(
            image_paths=[file_io.Path(p) for p, _ in self._depth_generate_queue]
        )
        to = [file_io.Path(p) for _, p in self._depth_generate_queue]

        @find_executable
        def closure(n: int):
            scale_factor = 1 / (2**n)
            logger.info("Resizing images by %f", scale_factor)

            # Run the Huggingface Transformers pipeline on the dataset
            with tqdm(desc="Depth generation", total=len(to)) as pbar:
                for image, depth_path in zip(
                    ds,
                    to,
                    strict=True,
                ):
                    # NOTE: we have to resize the depth map ("predicted_depth") **tensor**
                    # to the original size, and then save it. It is not possible to directly
                    # save the output PIL image ("depth"), because this has been quantized to
                    # 8-bit for visualization purposes.
                    image = to_tensor(image)
                    *_, height, width = image.shape
                    image_min_size = min(height, width, 1024)
                    image = resize_image(image, [int(image_min_size * scale_factor)])
                    image = to_pil_image(image)

                    pred = self._depth_pipeline(image)
                    depth = (
                        nn.functional.interpolate(
                            pred["predicted_depth"].unsqueeze(0),
                            size=tuple(map(int, (height, width))),
                            mode="nearest-exact",
                        )
                        .squeeze(0)
                        .as_subclass(DepthMap)
                    )
                    depth.save(depth_path, format=self._depth_format)

                    pbar.update(1)

        closure()

        # Queue was handled, clear it
        self._depth_generate_queue.clear()
        del self._depth_pipeline

    def run_depth_generator(
        self, image: torch.Tensor | pil_image.Image | np.ndarray
    ) -> torch.Tensor:
        """Run the depth generator on a single image."""
        from torchvision.transforms.v2.functional import to_pil_image

        if isinstance(image, (torch.Tensor, np.ndarray)):
            image = to_pil_image(image)
        elif isinstance(image, pil_image.Image):
            pass
        else:
            msg = f"Expected PIL image or tensor, got {type(image)}!"
            raise TypeError(msg)
        return self._depth_pipeline(image)["predicted_depth"]  # type: ignore
