"""
YT Collision
"""

from __future__ import annotations

import torchvision.transforms.v2 as transforms
from detectron2.evaluation import DatasetEvaluators

import unipercept as up
import unipercept.data.sets.cityscapes
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

__all__ = ["data"]

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(up.data.sets.YTCollision)(
                split="train",
                queue_fn=L(unipercept.data.collect.ExtractIndividualFrames)(
                    required_capture_sources={"image"},
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
                num_workers=4,
            ),
        ),
        "test": L(unipercept.data.DataLoaderFactory)(
            dataset=L(unipercept.data.sets.YTCollision)(
                split="val",
                queue_fn=L(unipercept.data.collect.ExtractIndividualFrames)(
                    required_capture_sources={"image"},
                ),
            ),
            actions=[
                L(unipercept.data.ops.CloneOp)(),
                L(unipercept.data.ops.TorchvisionOp)(transforms=[L(transforms.Resize)(size=1024, antialias=True)]),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=1,
            ),
        ),
    },
    evaluator=L(DatasetEvaluators)(evaluators=[]),
)
