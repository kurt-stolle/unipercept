"""Configuration builder for trackers."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
import unimodels.multidvps.keys as keys
import unitrack
from tensordict.nn import TensorDictModule

import unipercept as up

NESTED_OBJ = ("outputs", "predictions", keys.OUT_OBJECT)


def _define_simple_field(name: str, *, key: tuple[str, ...]) -> TensorDictModule:
    """
    Quick macro for defining a field that selects a tensor from the input dict.
    """
    return TensorDictModule(module=nn.Identity(), in_keys=key, out_keys=[name])


def build_embedding_tracker() -> up.nn.layers.tracking.StatefulTracker:
    """
    Builds a tracker that uses **only** the appearance embedding.
    """

    return up.nn.layers.tracking.StatefulTracker(
        tracker=unitrack.MultiStageTracker(
            fields=[
                _define_simple_field("embeddings", key=(*NESTED_OBJ, "kernels", keys.KEY_REID)),
                _define_simple_field("scores", key=(*NESTED_OBJ, "scores")),
                _define_simple_field("categories", key=(*NESTED_OBJ, "categories")),
            ],
            stages=[
                unitrack.stages.Association(
                    cost=unitrack.costs.Cosine(field="embeddings"),
                    assignment=unitrack.assignment.Jonker(threshold=0.2),
                ),
            ],
        ),
        memory=unitrack.TrackletMemory(
            states=dict(
                scores=unitrack.states.Value(torch.float),
                categories=unitrack.states.Value(torch.long),
                embeddings=unitrack.states.Value(torch.float),
            ),
        ),
    )


def build_depth_guided_tracker_from_metadata(name: str) -> up.nn.layers.tracking.StatefulTracker:
    info: up.data.sets.Metadata = up.get_info(name)
    return up.nn.layers.tracking.StatefulTracker(
        tracker=unitrack.MultiStageTracker(
            fields=[
                _define_simple_field("embeddings", key=(*NESTED_OBJ, "kernels", keys.KEY_REID)),
                _define_simple_field("scores", key=(*NESTED_OBJ, "scores")),
                _define_simple_field("categories", key=(*NESTED_OBJ, "categories")),
                (TensorDictModule)(
                    module=(up.nn.layers.projection.DepthProjection)(
                        max_depth=info.depth_max,
                    ),
                    in_keys=[
                        ("things", "masks", "logits"),
                        ("things", "depths", "means"),
                        ("camera",),
                    ],
                    out_keys=["projections"],
                ),
            ],
            stages=[
                unitrack.stages.Association(
                    cost=unitrack.costs.Cosine(field="association_embeddings"),
                    assignment=unitrack.assignment.Jonker(threshold=0.2),
                ),
                unitrack.stages.Association(
                    cost=unitrack.costs.Distance(field="projections"),
                    assignment=unitrack.assignment.Jonker(threshold=0.3),
                ),
            ],
        ),
        memory=unitrack.TrackletMemory(
            states=dict(
                scores=unitrack.states.Value(torch.float),
                categories=unitrack.states.Value(torch.long),
                kernels_mask=unitrack.states.Value(torch.float),
                projections=unitrack.states.Value(torch.float),
            ),
        ),
    )
