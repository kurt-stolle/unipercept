"""
This is a toy segmenter that uses the unipercept library to segment PASCAL VOC images.
"""

from __future__ import annotations

from torchvision.transforms import v2 as transforms
from unipercept.config import call as L
from unipercept.config import make_dict, make_set
from unipercept.data import DataLoaderFactory, ops, sets
from unipercept.engine import (
    Engine,
    EngineParams,
    EvaluationSuite,
    Interval,
    OptimizerFactory,
    TrainingStage,
    callbacks,
)
from unipercept.evaluators import PanopticEvaluator
from unipercept.integrations.wandb_integration import WandBCallback
from unipercept.model import toys
from unipercept.nn import backbones, layers, wrappers

__all__ = ["ENGINE", "MODEL", "HYPERPARAMS"]

HYPERPARAMS = L(make_dict)(
    dataset_name="pascal-voc",
)

ENGINE = L(Engine)(
    params=L(EngineParams)(
        project_name="unipercept_toys",
        eval_interval=L(Interval)(amount=1000, unit="steps"),
        logging_steps=20,
        trackers=L(make_set)(items=["wandb"]),
    ),
    callbacks=[
        L(callbacks.FlowCallback)(),
        L(callbacks.ProgressCallback)(),
        L(callbacks.GradientClippingCallback)(
            max_norm=5,
        ),
        L(WandBCallback)(
            watch_model=None,
        ),
    ],
    evaluators={
        "pascal-voc/val": L(EvaluationSuite)(
            enabled=True,
            loader=L(DataLoaderFactory.with_inference_defaults)(
                dataset=L(sets.pascal_voc.PascalVOCDataset)(
                    split="val",
                    year="2012",
                ),
                actions=[
                    L(ops.PadToDivisible)(divisor=512),
                ],
            ),
            handlers=[
                L(PanopticEvaluator.from_metadata)(
                    name="${HYPERPARAMS.dataset_name}",
                ),
            ],
        ),
    },
    stages=[
        L(TrainingStage)(
            loader=L(DataLoaderFactory.with_training_defaults)(
                dataset=L(sets.pascal_voc.PascalVOCDataset)(
                    split="train",
                    year="2012",
                ),
                actions=[
                    L(ops.TorchvisionOp)(
                        transforms=[
                            L(transforms.RandomCrop)(
                                size=(512, 512),
                                pad_if_needed=True,
                                fill=L(ops.get_fill_values)(),
                            ),
                            L(transforms.RandomHorizontalFlip)(),
                        ],
                    ),
                ],
            ),
            batch_size=8,
            iterations=L(Interval)(amount=100_000, unit="steps"),
            optimizer=L(OptimizerFactory)(
                opt="adamw",
                pkg="schedule_free",
                lr=1e-4,
                warmup_steps=2000,
            ),
        ),
    ],
)

MODEL = L(toys.Segmenter)(
    backbone=L(backbones.fpn.FeaturePyramidNetwork)(
        bottom_up=L(wrappers.freeze_parameters)(
            module=L(backbones.timm.TimmBackbone)(name="resnet50d")
        ),
        in_features=["ext_2", "ext_3", "ext_4"],
        out_channels=64,
        norm=layers.norm.GroupNorm32,
        extra_blocks=None,
    ),
    dims=64,
    classes=L(sets.catalog.get_info_at)(
        query="${HYPERPARAMS.dataset_name}", key="stuff_amount"
    ),
)
