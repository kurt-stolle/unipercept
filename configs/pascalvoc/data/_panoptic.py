"""
CityscapesDVPS
"""

from __future__ import annotations

import torchvision.transforms.v2 as transforms
from detectron2.evaluation import DatasetEvaluators
from uniutils.config._lazy import bind as B
from uniutils.config._lazy import call as L

import unipercept as up

__all__ = ["data"]


data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(up.data.sets.PascalVOC)(
                split="train",
                download=True,
                year="2012",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomHorizontalFlip)(),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=32,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(up.data.sets.PascalVOC)(
                split="val",
                download=True,
                year="2012",
                all=True,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(batch_size=1),
        ),
    },
    evaluator=L(DatasetEvaluators)(evaluators=[]),
)
