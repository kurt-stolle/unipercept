from __future__ import annotations

import typing as T
from logging import warn
from pathlib import Path

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import multidvps
from uniutils.config import infer_project_name, infer_session_name
from uniutils.config._lazy import bind as B
from uniutils.config._lazy import call as L
from uniutils.config._lazy import use_activation, use_norm

import unipercept as up

from .data._panoptic import data

__all__ = ["model", "data", "trainer"]

_MULTI_DIMS = 256
_MASK_DIMS = 128
_DEPTH_DIMS = 96
_REID_DIMS = 64
_INFO = up.data.read_info(data)

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name=infer_project_name(__file__),
        session_name=infer_session_name(),
        train_epochs=50,
        eval_epochs=1,
        save_epochs=1,
    ),
    optimizer=L(up.trainer._optimizer.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.trainer._scheduler.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.trainer.callbacks.FlowCallback, up.trainer.callbacks.ProgressCallback],
)

model = B(multidvps.MultiDVPS)(
    center_top_num=200,
    weighted_num=7,
    center_thres=0.01,
    sem_thres=0.13,
    panoptic_overlap_thrs=0.50,
    panoptic_stuff_limit=4096,
    panoptic_inst_thrs=0.23,
    common_stride=4,
    id_map_stuff=_INFO.stuff_embeddings,
    id_map_thing=_INFO.thing_embeddings,
    stuff_all_classes=_INFO.stuff_all_classes,
    stuff_with_things=_INFO.stuff_with_things,
    backbone=L(up.modeling.backbones.fpn.FeaturePyramidBackbone)(
        base=L(up.modeling.backbones.timm.TimmBackbone)(name="resnet50"),
        in_features=["ext.2", "ext.3", "ext.4", "ext.5"],
        routing=L(up.modeling.backbones.fpn.build_quad_routing)(num_levels=7),
        out_channels=128,
        num_hidden=1,
    ),
    detector=L(multidvps.modules.Detector)(
        in_features=[f"fpn.{i}" for i in (3, 4, 5, 6, 7)],
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
            stuff_channels=_INFO.stuff_amount,
            thing_channels=_INFO.thing_amount,
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
                multidvps.KEY_MULTI: L(multidvps.modules.Encoder)(
                    in_channels=T.cast(int, "${.....backbone.out_channels}"),
                    out_channels=_MULTI_DIMS,
                    num_convs=3,
                    deform=True,
                    coord=L(up.modeling.layers.CoordCat2d)(),
                    norm=use_norm("GN"),
                )
            },
        ),
    ),
    feature_encoder=L(multidvps.modules.FeatureEncoder)(
        merger=L(up.modeling.layers.SemanticMerge)(
            input_shape={
                f: L(ShapeSpec)(stride=s, channels=T.cast(int, "${.....backbone.out_channels}"))
                for f, s in zip(["fpn.2", "fpn.3", "fpn.4", "fpn.5"], [4, 8, 16, 32])
            },
            in_features=[f"fpn.{i}" for i in (2, 3, 4, 5)],
            out_channels=256,
            common_stride=4,
        ),
        heads={
            multidvps.KEY_MASK: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=_MASK_DIMS,
                num_convs=3,
                deform=True,
                norm=use_norm("GN"),
                coord=L(up.modeling.layers.CoordCat2d)(),
            ),
            multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${...merger.out_channels}"),
                out_channels=_DEPTH_DIMS,
                num_convs=3,
                deform=True,
                norm=use_norm("GN"),
                coord=L(up.modeling.layers.CoordCat2d)(),
            ),
        },
    ),
    fusion_thing=L(multidvps.modules.ThingFusion)(
        key=multidvps.KEY_MULTI,
        dims=_MULTI_DIMS,
        fusion_threshold=0.5,
        mapping={
            multidvps.KEY_MASK: L(up.modeling.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=_MASK_DIMS,
            ),
            multidvps.KEY_DEPTH: L(up.modeling.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=_DEPTH_DIMS,
            ),
            multidvps.KEY_REID: L(up.modeling.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=_REID_DIMS,
            ),
        },
    ),
    fusion_stuff=L(multidvps.modules.StuffFusion)(
        key=multidvps.KEY_MULTI,
        dims=_MULTI_DIMS,
        mapping={
            multidvps.KEY_MASK: L(up.modeling.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=_MASK_DIMS,
            ),
            multidvps.KEY_DEPTH: L(up.modeling.layers.MapMLP)(
                in_channels=T.cast(int, "${...dims}"),
                out_channels=_DEPTH_DIMS,
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
        kernel_dims=[_DEPTH_DIMS, _MASK_DIMS],
        max_depth=_INFO.depth_max,
    ),
    inference_pipeline=L(multidvps.logic.inference.InferencePipeline)(
        inst_thres=0.45,
    ),
    training_pipeline=L(multidvps.logic.training.TrainingPipeline)(
        stuff_channels=_INFO.stuff_amount,
        loss_location_weight=[5.0, 4.0],  # [thing, stuff]
        loss_location_thing=L(up.modeling.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_location_stuff=L(up.modeling.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
        ),
        loss_segment_thing=L(up.modeling.losses.WeightedLoss)(
            loss=L(up.modeling.losses.WeightedThingDiceLoss)(),
            weight=4.0,
        ),  # type: ignore
        loss_segment_stuff=L(up.modeling.losses.WeightedLoss)(
            loss=L(up.modeling.losses.WeightedStuffDiceLoss)(),
            weight=4.0,
        ),  # type: ignore
        loss_depth_means=L(up.modeling.losses.WeightedLoss)(
            loss=L(up.modeling.losses.DepthLoss)(),
            weight=1.0,
        ),
        loss_depth_values=L(up.modeling.losses.WeightedLoss)(
            loss=L(up.modeling.losses.DepthLoss)(),
            weight=1.0,
        ),
        truth_generator=L(multidvps.modules.supervision.TruthGenerator)(
            ignore_val=-1,
            common_stride=4,
            thing_embeddings=_INFO.thing_embeddings,
            stuff_embeddings=_INFO.stuff_embeddings,
            stuff_all_classes=_INFO.stuff_all_classes,
            stuff_with_things=_INFO.stuff_with_things,
            min_overlap=0.7,
            gaussian_sigma=3,
            label_divisor=_INFO.label_divisor,
        ),
    ),
)
