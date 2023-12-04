"""
Use PascalVOC for testing on small dataset.
"""

from __future__ import annotations

import typing as T
from logging import warn
from pathlib import Path

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import multidvps

import unipercept as up
from unipercept.config import get_project_name, get_session_name
from unipercept.config._lazy import bind as B
from unipercept.config._lazy import call as L
from unipercept.config._lazy import use_activation, use_norm

from .data._panoptic import DATASET_INFO, data

__all__ = ["model", "data", "engine"]

MULTI_DIMS = 256
MASK_DIMS = 128
DEPTH_DIMS = 96
REID_DIMS = 64

engine = B(up.engine.Engine)(
    config=L(up.engine.params.EngineParams)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(),
        train_batch_size=8,
        train_epochs=10,
        eval_steps=20,
        save_epochs=1,
    ),
    optimizer=L(up.engine.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.engine.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.engine.callbacks.FlowCallback, up.engine.callbacks.ProgressCallback],
)

model = B(multidvps.MultiDVPS)(
    weighted_num=7,
    common_stride=4,
    id_map_stuff=DATASET_INFO.stuff_embeddings,
    id_map_thing=DATASET_INFO.thing_embeddings,
    stuff_all_classes=DATASET_INFO.stuff_all_classes,
    stuff_with_things=DATASET_INFO.stuff_with_things,
    backbone=L(up.nn.backbones.fpn.FeaturePyramidBackbone)(
        base=L(up.nn.backbones.timm.TimmBackbone)(name="resnet50"),
        in_features=["ext.2", "ext.3", "ext.4", "ext.5"],
        routing=L(up.nn.backbones.fpn.build_default_routing)(num_levels=5),
        out_channels=128,
        num_hidden=1,
    ),
    detector=L(multidvps.modules.Detector)(
        in_features=[f"fpn.{i}" for i in (3, 4, 5)],
        localizer=L(multidvps.modules.Localizer)(
            encoder=L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${....backbone.out_channels}"),
                out_channels=256,
                num_convs=3,
                deform=True,
                coord=None,
                norm=use_norm("GN"),
                activation=use_activation(nn.GELU),
            ),
            stuff_channels=DATASET_INFO.stuff_amount,
            thing_channels=DATASET_INFO.thing_amount,
        ),
        kernelizer=L(multidvps.modules.Kernelizer)(
            heads={
                # multidvps.KEY_MASK: L(multidvps.modules.Encoder)(
                #     in_channels="${.....backbone.out_channels}",
                #     out_channels="${.....maskifier_thing.kernel_dims}",
                #     num_convs=3,
                #     deform=False,
                #     coord=L(layers.CoordCat2d)(),
                #     norm=use_norm("GN"),
                # ),
                # multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                #     in_channels="${.....backbone.out_channels}",
                #     out_channels="${.....depth_mapper.kernel_dims}",
                #     num_convs=3,
                #     deform=True,
                #     coord=L(layers.CoordCat2d)(),
                #     norm=use_norm("GN"),
                # ),
                multidvps.KEY_SEMANTIC: L(multidvps.modules.Encoder)(
                    in_channels=T.cast(int, "${.....backbone.out_channels}"),
                    out_channels=MULTI_DIMS,
                    num_convs=3,
                    deform=True,
                    coord=L(up.nn.layers.CoordCat2d)(),
                    norm=use_norm("GN"),
                )
            },
        ),
    ),
    feature_encoder=L(multidvps.modules.FeatureEncoder)(
        merger=L(up.nn.layers.SemanticMerge)(
            input_shape={
                f: L(ShapeSpec)(stride=s, channels=T.cast(int, "${.....backbone.out_channels}"))
                for f, s in zip(["fpn.1", "fpn.2", "fpn.3", "fpn.4"], [4, 8, 16, 32])
            },
            in_features=[f"fpn.{i}" for i in (1, 2, 3, 4)],
            out_channels=256,
            common_stride=4,
        ),
        heads={
            multidvps.KEY_MASK: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=MASK_DIMS,
                num_convs=3,
                deform=True,
                norm=use_norm("GN"),
                coord=L(up.nn.layers.CoordCat2d)(),
            ),
            multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=DEPTH_DIMS,
                num_convs=3,
                deform=True,
                norm=use_norm("GN"),
                coord=L(up.nn.layers.CoordCat2d)(),
            ),
        },
    ),
    fusion_thing=L(multidvps.modules.ThingFusion)(
        key=multidvps.KEY_SEMANTIC,
        dims=MULTI_DIMS,
        fusion_threshold=0.5,
        mapping={
            multidvps.KEY_MASK: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=MASK_DIMS,
            ),
            multidvps.KEY_DEPTH: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=DEPTH_DIMS,
            ),
            multidvps.KEY_REID: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=REID_DIMS,
            ),
        },
    ),
    fusion_stuff=L(multidvps.modules.StuffFusion)(
        key=multidvps.KEY_SEMANTIC,
        dims=MULTI_DIMS,
        mapping={
            multidvps.KEY_MASK: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=MASK_DIMS,
            ),
            multidvps.KEY_DEPTH: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=DEPTH_DIMS,
            ),
        },
    ),
    maskifier_thing=L(multidvps.modules.MaskHead)(
        key=multidvps.KEY_MASK,
    ),
    maskifier_stuff=L(multidvps.modules.MaskHead)(
        key=multidvps.KEY_MASK,
    ),
    depth_mapper=L(multidvps.modules.DepthHead)(
        feature_key=multidvps.KEY_DEPTH,
        kernel_keys=[multidvps.KEY_DEPTH, multidvps.KEY_MASK],
        kernel_dims=[DEPTH_DIMS, MASK_DIMS],
        max_depth=DATASET_INFO.depth_max,
    ),
    inference_pipeline=L(multidvps.logic.inference.InferencePipeline)(
        center_thres=0.01,
        sem_thres=0.13,
        center_top_num=200,
        panoptic_overlap_thrs=0.50,
        panoptic_stuff_limit=4096,
        panoptic_inst_thrs=0.23,
        inst_thres=0.45,
    ),
    training_pipeline=L(multidvps.logic.training.TrainingPipeline)(
        stuff_channels=DATASET_INFO.stuff_amount,
        loss_location_weight=[5.0, 4.0],  # [thing, stuff]
        loss_location_thing=L(up.nn.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_location_stuff=L(up.nn.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_segment_thing=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.WeightedThingDiceLoss)(),
            weight=4.0,
        ),  # type: ignore
        loss_segment_stuff=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.WeightedStuffDiceLoss)(),
            weight=4.0,
        ),  # type: ignore
        loss_depth_means=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.DepthLoss)(),
            weight=1.0,
        ),
        loss_depth_values=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.DepthLoss)(),
            weight=1.0,
        ),
        truth_generator=L(multidvps.modules.supervision.TruthGenerator)(
            ignore_val=-1,
            common_stride=4,
            thing_embeddings=DATASET_INFO.thing_embeddings,
            stuff_embeddings=DATASET_INFO.stuff_embeddings,
            stuff_all_classes=DATASET_INFO.stuff_all_classes,
            stuff_with_things=DATASET_INFO.stuff_with_things,
            min_overlap=0.7,
            gaussian_sigma=3,
            label_divisor=DATASET_INFO.label_divisor,
        ),
        loss_reid=L(up.nn.losses.WeightedLoss)(
            loss=L(nn.TripletMarginWithDistanceLoss)(
                distance_function=L(nn.CosineSimilarity)(),
                margin=0.2,
            ),
            weight=1.0,
        ),
        loss_dgp=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.DGPLoss)(),
            weight=1.0,
        ),
        loss_pgt=L(up.nn.losses.WeightedLoss)(
            loss=L(up.nn.losses.PGTLoss)(),
            weight=1.0,
        ),
    ),
)
