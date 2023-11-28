from __future__ import annotations

import torchvision.transforms.v2 as transforms

import unipercept as up
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

__all__ = ["data"]

DATASET_NAME = "kitti-360"
DATASET_CLASS: type[up.data.sets.cityscapes.CityscapesVPSDataset] = L(up.get_dataset)(name=DATASET_NAME)
DATASET_INFO: up.data.sets.Metadata = up.data.sets.get_info(DATASET_NAME)

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="train",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomResize)(min_size=512, max_size=1024, antialias=True),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                        L(transforms.RandomHorizontalFlip)(),
                        L(transforms.RandomPhotometricDistort)(),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(DATASET_CLASS)(
                split="val",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
                ),
            ),
            actions=[
                L(up.data.ops.CloneOp)(),
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=512, antialias=True),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="inference"),
            config=L(up.data.DataLoaderConfig)(
                drop_last=False,
            ),
        ),
    }
)
