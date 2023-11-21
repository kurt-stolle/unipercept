"""
SemKITTI (DVPS)
"""

from __future__ import annotations

import evaluators
import torchvision.transforms.v2 as transforms

import unipercept as up
import unipercept.data.sets.cityscapes
from unipercept.utils.config import bind as B
from unipercept.utils.config import call as L

__all__ = ["data"]


data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(up.data.sets.SemKITTIDataset)(
                split="train",
                pseudo=True,
                queue_fn=L(unipercept.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources={"image", "depth", "panoptic"},
                ),
            ),
            actions=[
                L(unipercept.data.ops.CloneOp)(),
                L(unipercept.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.AutoAugment)(),
                        L(transforms.RandomResizedCrop)(size=[512, 1024]),
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
            dataset=L(unipercept.data.sets.SemKITTIDataset)(
                split="val",
                queue_fn=L(unipercept.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "depth", "panoptic"},
                ),
            ),
            actions=[
                L(unipercept.data.ops.CloneOp)(),
                L(unipercept.data.ops.TorchvisionOp)(transforms=[L(transforms.Resize)(size=1024)]),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=1, drop_last=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
            ),
        ),
    },
    evaluator=L(DatasetEvaluators)(evaluators=[]),
)
