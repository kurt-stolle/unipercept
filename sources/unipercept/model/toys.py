r"""
Toy models for use in tutorials and testing.
"""

from __future__ import annotations

import typing as T

import numpy as np
import typing_extensions as TX
from torch import Tensor, nn, stack, zeros_like

from unipercept.data.tensors import PanopticMap
from unipercept.nn.activations import ActivationSpec, InplaceReLU, get_activation
from unipercept.nn.backbones import BackboneFeatureInfo
from unipercept.nn.layers.conv import Separable2d
from unipercept.nn.norms import GroupNorm32, NormSpec

from ._base import InputData, ModelBase, ModelOutput


class _SegmenterEncoder(nn.Module):
    """
    Segmenter head that takes a FPN and produces a feature space
    """

    common_stride: T.Final[int]
    in_features: T.Final[list[str]]

    def __init__(
        self,
        in_features: T.Mapping[str, BackboneFeatureInfo],
        common_stride: int,
        out_channels: int,
        norm: NormSpec = GroupNorm32,
        activation: ActivationSpec = InplaceReLU,
    ):
        super().__init__()

        self.in_features = list(in_features.keys())
        self.common_stride = int(common_stride)
        self.out_channels = out_channels

        feature_strides = {k: T.cast(int, v.stride) for k, v in in_features.items()}
        feature_channels = {k: T.cast(int, v.channels) for k, v in in_features.items()}

        self.scale_heads = nn.ModuleList()
        for in_feature in self.in_features:
            head_ops = nn.Sequential()
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            in_channels = feature_channels[in_feature]
            if feature_strides[in_feature] != self.common_stride:
                scale_factor = 2 * head_length
            else:
                scale_factor = 1

            conv = Separable2d.with_norm_activation(
                in_channels,
                out_channels * scale_factor**2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=norm is None,
                norm=norm,
                activation=get_activation(activation),
            )

            head_ops.add_module("conv", conv)
            if scale_factor > 0:
                head_ops.add_module("shuf", nn.PixelShuffle(scale_factor))
            self.scale_heads.append(head_ops)

    def __len__(self) -> int:
        return len(self.in_offsets)

    @TX.override
    def forward(self, features: dict[str, Tensor]) -> Tensor:
        return stack(
            [
                head(features[self.in_features[i]])
                for i, head in enumerate(self.scale_heads)
            ],
            dim=-1,
        ).sum(dim=-1)

    if T.TYPE_CHECKING:
        __call__ = forward


class _SegmenterCriterion(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, smooth=1.0):
        super().__init__()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth

    @TX.override
    def forward(self, logits: Tensor, ground_truth: Tensor) -> dict[str, Tensor]:
        n_batch, n_classes, h, w = logits.shape

        # Cross-entropy without ignored labels
        target = nn.functional.one_hot(ground_truth + 1, num_classes=n_classes + 1)
        target = target[..., 1:]  # Remove ignored label that we added by shifting by 1
        target = target.float().permute(0, 3, 1, 2)

        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(n_batch, n_classes, -1),
            target.reshape(n_batch, n_classes, -1),
            reduction="none",
        ).mean()

        return {
            "cross_entropy": self.weight_ce * ce_loss,
        }


class Segmenter(ModelBase):
    def __init__(self, backbone: nn.Module, dims: int, classes: int):
        super().__init__()
        self.backbone = backbone
        self.encoder = _SegmenterEncoder(
            in_features=backbone.get_backbone_features(),
            common_stride=4,
            out_channels=dims,
        )
        self.head = nn.Sequential(
            Separable2d.with_norm_activation(
                dims,
                dims,
                kernel_size=7,
                stride=1,
                padding=7 // 2,
                norm=GroupNorm32,
                activation=InplaceReLU,
            ),
            Separable2d.with_norm_activation(
                dims,
                dims,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                norm=GroupNorm32,
                activation=InplaceReLU,
            ),
            Separable2d.with_norm_activation(
                dims,
                dims,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                norm=GroupNorm32,
                activation=InplaceReLU,
            ),
            nn.Conv2d(dims, classes, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            ),
        )
        self.criterion = _SegmenterCriterion()

    @TX.override
    def select_inputs(self, data: InputData, device: T.Any) -> tuple[T.Any, ...]:
        images = data.captures.images.flatten(0, 1).to(device, non_blocking=True)
        if self.training:
            return images, data.captures.segmentations.flatten(0, 1).as_subclass(
                PanopticMap
            ).get_semantic_map().to(device, non_blocking=True)
        return images, None

    @TX.override
    def forward(self, images: Tensor, segmentations: Tensor) -> ModelOutput:
        features = self.backbone(images)
        logits = self.encoder(features)
        logits = self.head(logits)

        if self.training:
            losses = self.criterion(logits, segmentations)
            predictions = None
        else:
            preds = logits.sigmoid()
            # sem_map = preds.argmax(dim=1)
            scores, sem_map = preds.max(dim=1)
            sem_map[scores < 0.01] = -1
            ins_map = zeros_like(sem_map)

            losses = None
            predictions = {
                "segmentations": PanopticMap.from_parts(sem_map, ins_map),
            }

        return ModelOutput(
            losses=losses,
            predictions=predictions,
        )

    if T.TYPE_CHECKING:
        __call__ = forward
