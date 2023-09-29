"""
CityscapesDVPS
"""

from __future__ import annotations

import torchvision.transforms.v2 as transforms
import unipercept.data.sets.cityscapes
from detectron2.evaluation import DatasetEvaluators
from uniutils.config._lazy import bind as B
from uniutils.config._lazy import call as L

import unipercept as up

__all__ = ["data"]


data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(up.data.sets.CityscapesVPS)(
                split="train",
                all=False,
                queue_fn=L(unipercept.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources={"image", "depth", "panoptic"},
                ),
            ),
            actions=[
                L(unipercept.data.ops.CloneOp)(),
                L(unipercept.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomResizedCrop)(size=[512, 1024], antialias=True),
                        L(transforms.RandomHorizontalFlip)(),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=32,
                drop_last=True,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2,
                persistent_workers=True,
            ),
        ),
        "test": L(unipercept.data.DataLoaderFactory)(
            dataset=L(unipercept.data.sets.CityscapesVPS)(
                split="val",
                all=True,
                queue_fn=L(unipercept.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "depth", "panoptic"},
                ),
            ),
            actions=[
                L(unipercept.data.ops.CloneOp)(),
                L(unipercept.data.ops.TorchvisionOp)(transforms=[L(transforms.Resize)(size=1024, antialias=True)]),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=1, drop_last=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
            ),
        ),
    },
    evaluator=L(DatasetEvaluators)(evaluators=[]),
)
