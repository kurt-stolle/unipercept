from __future__ import annotations

import evaluators
import torchvision.transforms.v2 as transforms

import unipercept as up
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

__all__ = ["data"]

DATASET_CLASS: type[up.data.sets.PerceptionDataset] = up.data.sets.pascal_voc.PascalVOCDataset
DATASET_INFO: up.data.sets.Metadata = DATASET_CLASS.read_info()

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="train",
                download=False,  # True,
                year="2012",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomHorizontalFlip)(),
                        L(transforms.Resize)(size=256, antialias=True),
                    ]
                ),
                L(up.data.ops.PseudoMotion)(frames=2, size=256),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(
                batch_size=4,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="val",
                download=False,  # True,
                year="2012",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=256, antialias=True),
                        L(transforms.CenterCrop)(size=[256, 256]),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(batch_size=1),
        ),
    }
)
