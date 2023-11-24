from __future__ import annotations

import typing as T
import torch
import torch.nn as nn
import unimodels.multidvps as multidvps
import unipercept as up
from unipercept.utils.config import bind as B
from unipercept.utils.config import call as L
from unipercept.utils.config import (
    get_session_name,
)

from ._dataset import DATASET_INFO, DATASET_NAME, data

__all__ = ["model", "data", "trainer"]

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name="multidvps",
        session_name=get_session_name(__file__),
        train_batch_size=8,
        train_epochs=200,
        infer_batch_size=1,
        eval_steps=5_000,
        save_steps=5_000,
        logging_steps=100,
    ),
    optimizer=L(up.trainer.OptimizerFactory)(
        opt="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4, weight_decay_norm=0.0, weight_decay_bias=1e-8
    ),
    scheduler=L(up.trainer.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
        cooldown_epochs=0,
    ),
    callbacks=[
        L(up.trainer.callbacks.FlowCallback)(),
        L(up.trainer.callbacks.ProgressCallback)(),
    ],
    evaluators=[
        L(up.evaluators.DepthEvaluator.from_metadata)(name=DATASET_NAME),
        L(up.evaluators.PanopticEvaluator.from_metadata)(name=DATASET_NAME),
    ],
)

model = B(multidvps.MultiDVPS.from_metadata)(
    dataset_name=DATASET_NAME,
    weighted_num=7,
    common_stride=4,
    backbone=L(up.nn.backbones.fpn.FeaturePyramidNetwork)(
        bottom_up=L(up.nn.backbones.timm.TimmBackbone)(name="resnet50d"),
        in_features=["ext.2", "ext.3", "ext.4", "ext.5"],
        out_channels=128,
        norm=up.nn.layers.norm.LayerNormCHW,
        extra_blocks=L(up.nn.backbones.fpn.LastLevelP6P7)(
            in_channels="${..out_channels}",
            out_channels="${..out_channels}",
        ),
        freeze=True,
    ),
    detector=L(multidvps.modules.Detector)(
        in_features=[f"fpn.{i}" for i in (3, 4, 5, 6)],
        localizer=L(multidvps.modules.Localizer)(
            encoder=L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${model.backbone.out_channels}"),
                out_channels=128,
                num_convs=3,
                deform=True,
                coord=L(up.nn.layers.CoordCat2d)(),
                norm=up.nn.layers.norm.GroupNormCG,
                activation=nn.GELU,
            ),
            squeeze_excite=L(up.nn.layers.SqueezeExcite2d)(
                channels=T.cast(int, "${..encoder.out_channels}"),
            ),
            stuff_channels=DATASET_INFO.stuff_amount,
            thing_channels=DATASET_INFO.thing_amount,
        ),
        kernelizer=L(multidvps.modules.Kernelizer)(
            heads={
                multidvps.KEY_GEOMETRY: L(multidvps.modules.GeometryEncoder)(
                    encoder=L(multidvps.modules.Encoder)(
                        in_channels=T.cast(int, "${model.backbone.out_channels}"),
                        out_channels=32,
                        num_convs=3,
                        groups=1,
                        deform=False,
                        coord=L(up.nn.layers.CoordCat2d)(),
                        norm=up.nn.layers.norm.LayerNormCHW,
                    ),
                    out_channels=2,
                ),
                multidvps.KEY_SEMANTIC: L(multidvps.modules.Encoder)(
                    in_channels=T.cast(int, "${model.backbone.out_channels}"),
                    out_channels=T.cast(int, "${model.kernel_mapper.input_dims}"),
                    num_convs=3,
                    groups=1,
                    deform=True,
                    coord=L(up.nn.layers.CoordCat2d)(),
                    norm=up.nn.layers.norm.LayerNormCHW,
                ),
            },
        ),
    ),
    feature_encoder=L(multidvps.modules.FeatureEncoder)(
        merger=L(up.nn.layers.merge.SemanticMerge)(
            input_shape={
                f: L(up.nn.backbones.BackboneFeatureInfo)(
                    stride=s, channels=T.cast(int, "${model.backbone.out_channels}")
                )
                for f, s in zip(["fpn.1", "fpn.2", "fpn.3", "fpn.4"], [4, 8, 16, 32])
            },
            in_features=["fpn.1", "fpn.2", "fpn.3", "fpn.4"],
            out_channels=128,
            common_stride=4,
        ),
        heads={
            multidvps.KEY_MASK: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=256,
                num_convs=3,
                deform=True,
                groups=8,
                norm=L(up.nn.layers.norm.GroupNormFactory)(num_groups=T.cast(int, "${..groups}")),
                coord=None,
            ),
            multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=64,
                num_convs=3,
                deform=True,
                groups=8,
                norm=L(up.nn.layers.norm.GroupNormFactory)(num_groups=T.cast(int, "${..groups}")),
                coord=None,
            ),
        },
    ),
    kernel_mapper=L(multidvps.modules.KernelMapper)(
        input_key=multidvps.KEY_SEMANTIC,
        input_dims=256,
        attention_heads=8,
        dropout=0.2,
        mapping={
            multidvps.KEY_MASK: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=f"${{model.feature_encoder.heads[{multidvps.KEY_MASK}].out_channels}}",
                dropout=0.0,
            ),
            multidvps.KEY_DEPTH: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=f"${{model.feature_encoder.heads[{multidvps.KEY_DEPTH}].out_channels}}",
                dropout=0.0,
            ),
            multidvps.KEY_REID: L(up.nn.layers.EmbedMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=64,
                dropout=0.5,
            ),
        },
    ),
    thing_embeddings=[multidvps.KEY_MASK, multidvps.KEY_DEPTH, multidvps.KEY_REID],
    stuff_embeddings=[multidvps.KEY_MASK, multidvps.KEY_DEPTH],
    fusion_thing=L(multidvps.modules.ThingFusion)(
        fusion_key=multidvps.KEY_MASK,
        fusion_threshold=0.95,
    ),
    fusion_stuff=L(multidvps.modules.StuffFusion)(),
    maskifier_thing=L(multidvps.modules.MaskHead)(
        key=multidvps.KEY_MASK,
    ),
    maskifier_stuff=L(multidvps.modules.MaskHead)(
        key=multidvps.KEY_MASK,
    ),
    depth_mapper=L(multidvps.modules.DepthHead)(
        feature_key=multidvps.KEY_DEPTH,
        geometry_key=multidvps.KEY_GEOMETRY,
        min_depth=2,
        max_depth=DATASET_INFO.depth_max,
    ),
    tracker=L(multidvps.trackers.build_embedding_tracker)(),
    inference_pipeline=L(multidvps.logic.inference.InferencePipeline)(
        center_thres=0.01,
        sem_thres=0.20,
        center_top_num=200,
        panoptic_overlap_thrs=0.50,
        panoptic_stuff_limit=2048,
        panoptic_inst_thrs=0.10,
        inst_thres=0.40,
    ),
    training_pipeline=L(multidvps.logic.training.TrainingPipeline.from_metadata)(
        name=DATASET_NAME,
        truth_generator=L(multidvps.modules.supervision.TruthGenerator.from_metadata)(
            name=DATASET_NAME,
            ignore_val=-1,
            common_stride=4,
            min_overlap=0.7,
            gaussian_sigma=3,
        ),
        loss_location_weight=[8.0, 6.0],  # [thing, stuff]
        loss_location_thing=L(up.nn.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_location_stuff=L(up.nn.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_segment_thing=L(up.nn.losses.WeightedThingDiceLoss)(scale=14.0),
        loss_segment_stuff=L(up.nn.losses.WeightedStuffDiceLoss)(scale=8.0),
        loss_depth_means=L(up.nn.losses.DepthLoss)(scale=1.0),
        loss_depth_values=L(up.nn.losses.DepthLoss)(scale=4.0),
        loss_reid=L(up.nn.losses.TripletMarginSimilarityLoss)(),
        loss_dgp=L(up.nn.losses.DGPLoss)(scale=0.5),
        loss_pgt=L(up.nn.losses.PGTLoss)(scale=0.5),
        loss_pgs=L(up.nn.losses.PGSLoss)(scale=0.5),
    ),
)
