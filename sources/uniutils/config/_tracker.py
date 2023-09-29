"""Configuration builder for trackers."""

from __future__ import annotations

import typing as T

from tensordict.nn import TensorDictModule
from torch.nn import Identity
from unipercept.modeling import layers
from unitrack import (
    MultiStageTracker,
    TrackletMemory,
    assignment,
    costs,
    stages,
    states,
)

from ._lazy import as_list, as_tuple
from ._lazy import call as L
from ._lazy import make_dict

if T.TYPE_CHECKING:
    from unipercept.data.types import Metadata

__all__ = ["build_tracker_weighted"]


def build_tracker_weighted(info: Metadata):
    return L(layers.tracking.StatefulTracker)(
        tracker=L(MultiStageTracker)(
            fields=[
                L(TensorDictModule)(
                    module=L(Identity)(),
                    in_keys=as_list(
                        as_tuple("things", "masks", "fused_kernels", "masks"),
                    ),
                    out_keys=as_list("kernels_mask"),
                ),
                L(TensorDictModule)(
                    module=L(Identity)(),
                    in_keys=as_list(as_tuple("things", "masks", "scores")),
                    out_keys=as_list("scores"),
                ),
                L(TensorDictModule)(
                    module=L(Identity)(),
                    in_keys=as_list(as_tuple("things", "masks", "categories")),
                    out_keys=as_list("categories"),
                ),
                L(TensorDictModule)(
                    module=L(layers.projection.DepthProjection)(
                        max_depth=info.depth_max,
                    ),
                    in_keys=as_list(
                        as_tuple("things", "masks", "logits"),
                        as_tuple("things", "depths", "means"),
                        as_tuple("camera"),
                    ),
                    out_keys=as_list("projections"),
                ),
            ],
            stages=[
                L(stages.Association)(
                    cost=L(costs.Cosine)(field="association_embeddings"),
                    assignment=L(assignment.Jonker)(threshold=0.2),
                ),
                L(stages.Association)(
                    cost=L(costs.Distance)(field="reverse_projections"),
                    assignment=L(assignment.Jonker)(threshold=0.3),
                ),
            ],
        ),
        memory=L(TrackletMemory)(
            states=L(make_dict)(
                scores=L(states.Value)(dtype="float"),
                categories=L(states.Value)(dtype="long"),
                kernels_mask=L(states.Value)(dtype="float"),
                projections=L(states.Value)(dtype="float"),
            ),
        ),
    )
