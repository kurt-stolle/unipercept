from __future__ import annotations

import typing as T

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import multidvps

import unipercept as up
from unipercept.utils.config import bind as B
from unipercept.utils.config import call as L
from unipercept.utils.config import (
    get_project_name,
    get_session_name,
    use_activation,
    use_norm,
)

from ._data import DATASET_INFO, DATASET_NAME, data

__all__ = ["model", "data", "trainer"]

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(__file__),
        train_batch_size=12,
        train_epochs=200,
        infer_batch_size=4,
        eval_epochs=4,
        save_epochs=4,
        logging_steps=50,
    ),
    optimizer=L(up.trainer.OptimizerFactory)(opt="sgd", lr=1e-2),
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

FEATURE_DIMS = 256
MULTI_DIMS = 256
MASK_DIMS = 256
DEPTH_DIMS = 32
REID_DIMS = 32

model = B(multidvps.MultiDVPS.from_metadata)(
    dataset_name=DATASET_NAME,
    weighted_num=7,
    common_stride=4,
    backbone=L(up.nn.backbones.fpn.FeaturePyramidBackbone)(
        base=L(up.nn.backbones.timm.TimmBackbone)(name="resnet50"),
        in_features=["ext.2", "ext.3", "ext.4", "ext.5"],
        routing=L(up.nn.backbones.fpn.build_pan_routing)(num_levels=6, weight_method="attn"),
        out_channels=88,
        num_hidden=2,
    ),
    detector=L(multidvps.modules.Detector)(
        in_features=[f"fpn.{i}" for i in (3, 4, 5, 6)],
        localizer=L(multidvps.modules.Localizer)(
            encoder=L(multidvps.modules.Encoder)(
                in_channels=T.cast(int, "${....backbone.out_channels}"),
                out_channels=256,
                num_convs=3,
                deform=True,
                coord=None,
                norm=up.nn.layers.norm.GroupNormCG,
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
                #     norm=up.nn.layers.norm.LayerNormCHW,
                # ),
                # multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                #     in_channels="${.....backbone.out_channels}",
                #     out_channels="${.....depth_mapper.kernel_dims}",
                #     num_convs=3,
                #     deform=True,
                #     coord=L(layers.CoordCat2d)(),
                #     norm=up.nn.layers.norm.LayerNormCHW,
                # ),
                multidvps.KEY_MULTI: L(multidvps.modules.Encoder)(
                    in_channels=T.cast(int, "${.....backbone.out_channels}"),
                    out_channels=MULTI_DIMS,
                    num_convs=3,
                    groups=32,
                    deform=True,
                    coord=L(up.nn.layers.CoordCat2d)(),
                    norm=up.nn.layers.norm.GroupNormCG,
                )
            },
        ),
    ),
    feature_encoder=L(multidvps.modules.FeatureEncoder)(
        merger=L(multidvps.modules.FeatureSelector)(
            name="fpn.1",
        ),
        shared_encoder=L(multidvps.modules.Encoder)(
            in_channels="${model.backbone.out_channels}",
            out_channels=FEATURE_DIMS,
            num_convs=1,
            deform=True,
            groups=1,
            norm=up.nn.layers.norm.GroupNormCG,
            coord=L(up.nn.layers.CoordCat2d)(),
        ),
        heads={
            multidvps.KEY_MASK: L(multidvps.modules.Encoder)(
                in_channels="${...shared_encoder.out_channels}",
                out_channels=MASK_DIMS,
                num_convs=2,
                deform=False,
                norm=up.nn.layers.norm.GroupNormCG,
                coord=None,
            ),
            multidvps.KEY_DEPTH: L(multidvps.modules.Encoder)(
                in_channels="${...shared_encoder.out_channels}",
                out_channels=DEPTH_DIMS,
                num_convs=2,
                deform=True,
                norm=up.nn.layers.norm.GroupNormCG,
                coord=None,
            ),
        },
    ),
    kernel_mapper=L(multidvps.modules.KernelMapper)(
        input_key=multidvps.KEY_MULTI,
        input_dims=MULTI_DIMS,
        attention_heads=4,
        dropout=0.0,
        mapping={
            multidvps.KEY_MASK: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=MASK_DIMS,
                dropout=0.0,
            ),
            multidvps.KEY_DEPTH: L(up.nn.layers.MapMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=DEPTH_DIMS,
                dropout=0.0,
            ),
            multidvps.KEY_REID: L(up.nn.layers.EmbedMLP)(
                in_channels=T.cast(int, "${...input_dims}"),
                out_channels=REID_DIMS,
                dropout=0.0,
            ),
        },
    ),
    thing_embeddings = [multidvps.KEY_MASK, multidvps.KEY_DEPTH, multidvps.KEY_REID],
    stuff_embeddings = [multidvps.KEY_MASK, multidvps.KEY_DEPTH],
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
        kernel_keys=[multidvps.KEY_DEPTH, multidvps.KEY_MASK],
        kernel_dims=[DEPTH_DIMS, MASK_DIMS],
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
            alpha=0.33,
            gamma=1.8,
        ),
        loss_location_stuff=L(up.nn.losses.SigmoidFocalLoss)(
            alpha=0.25,
            gamma=1.8,
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
