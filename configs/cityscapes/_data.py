from __future__ import annotations

import torchvision.transforms.v2 as transforms

import unipercept as up
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

__all__ = ["data"]

DATASET_NAME = "cityscapes/vps"
DATASET_CLASS: type[up.data.sets.cityscapes.CityscapesVPSDataset] = up.data.sets.get_dataset(DATASET_NAME)
DATASET_INFO: up.data.sets.Metadata = up.data.sets.get_info("cityscapes/vps")

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="train",
                all=False,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomResize)(min_size=512, max_size=1024 + 512, antialias=True),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                        L(transforms.RandomHorizontalFlip)(),
                        # L(transforms.RandomResizedCrop)(size=(512, 1024), scale=(0.5, 2), antialias=True),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(
                # num_workers=36,
                prefetch_factor=10,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="val",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=1024, antialias=True),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(
                pin_memory=True,
                drop_last=False,
                prefetch_factor=10,
                # num_workers=36,
            ),
        ),
    }
)
