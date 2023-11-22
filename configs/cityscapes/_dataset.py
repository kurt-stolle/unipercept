from __future__ import annotations

import torchvision.transforms.v2 as transforms

import unipercept as up
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

__all__ = ["data"]

DATASET_NAME = "cityscapes/vps"
DATASET_CLASS: type[up.data.sets.cityscapes.CityscapesVPSDataset] = L(up.data.sets.get_dataset)(name=DATASET_NAME)
DATASET_INFO: up.data.sets.Metadata = up.data.sets.get_info("cityscapes/vps")

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes/vps"))(
                split="train",
                all=False,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomResize)(min_size=512, max_size=1024 + 512, antialias=True),
                        L(transforms.RandomHorizontalFlip)(),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(),
        ),
        "pretrain": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes"))(
                split="train",
                queue_fn=L(up.data.collect.ExtractIndividualFrames)(
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.RandomResize)(min_size=512, max_size=1024, antialias=True),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                        L(transforms.RandomHorizontalFlip)(),
                    ]
                ),
                L(up.data.ops.PseudoMotion)(frames=1, size=(512, 1024))
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes/vps"))(
                split="val",
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources={"image", "panoptic"},
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=1024, antialias=True),
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
