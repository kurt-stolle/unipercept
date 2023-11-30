from __future__ import annotations

import torchvision.transforms.v2 as transforms

import unipercept as up
from unipercept.utils.config import bind as B
from unipercept.utils.config import call as L

__all__ = ["data"]

DATASET_NAME = "cityscapes/vps"
DATASET_INFO: up.data.sets.Metadata = up.data.sets.get_info("cityscapes/vps")

data = B(up.data.DataConfig)(
    loaders={
        "train": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes/vps"))(
                split="train",
                all=False,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.CenterCrop)(size=(1024 - 32, 2048 - 32)),
                        L(transforms.RandomResize)(min_size=512, max_size=1024, antialias=True),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                        L(transforms.RandomHorizontalFlip)(),
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
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=1024, antialias=True),
                        L(transforms.CenterCrop)(size=(1024 - 32, 2048 - 32)),
                        L(transforms.RandomResize)(min_size=512, max_size=2048, antialias=True),
                        L(transforms.RandomCrop)(size=(512, 1024), pad_if_needed=False),
                        L(transforms.RandomHorizontalFlip)(),
                        L(transforms.RandomPhotometricDistort)(),
                    ]
                ),
                L(up.data.ops.PseudoMotion)(frames=2, size=(512, 1024)),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(),
        ),
        "finetune": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes/vps"))(
                split="train",
                all=False,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=2,
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
                ),
            ),
            actions=[
                L(up.data.ops.TorchvisionOp)(
                    transforms=[
                        L(transforms.Resize)(size=1024, antialias=True),
                        L(transforms.RandomHorizontalFlip)(),
                    ]
                ),
            ],
            sampler=L(up.data.SamplerFactory)(sampler="training"),
            config=L(up.data.DataLoaderConfig)(),
        ),
        "test": L(up.data.DataLoaderFactory)(
            dataset=L(L(up.get_dataset)(name="cityscapes/vps"))(
                split="val",
                all=False,
                queue_fn=L(up.data.collect.GroupAdjacentTime)(
                    num_frames=1,
                    required_capture_sources=L(up.utils.config.make_set)(items=["image", "panoptic"]),
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
