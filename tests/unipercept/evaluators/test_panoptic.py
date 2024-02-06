from __future__ import annotations

import typing as T

import einops
import pytest
import torch
import typing_extensions as TX
from tensordict import TensorDict

from unipercept.evaluators import PRED_PANOPTIC, TRUE_PANOPTIC, PanopticEvaluator


def test_panoptic_evaluator(true_panoptic: torch.Tensor, 
                            pred_panoptic: torch.Tensor, true_info):
    """
    Test the panoptic evaluator.
    """

    from unipercept.data.tensors import PanopticMap

    sample_h, sample_w = true_panoptic.shape
    sample_amt = 3

    print(f"GT: {true_panoptic.unique().tolist()}")

    storage = TensorDict(
        {
            TRUE_PANOPTIC: einops.repeat(true_panoptic, "h w -> b h w", b=sample_amt),
            PRED_PANOPTIC: torch.cat(
                [
                    pred_panoptic.unsqueeze(0),
                    PanopticMap.from_parts(
                        torch.randint(
                            0,
                            max(true_info.stuff_ids | true_info.thing_ids),
                            (sample_amt - 1, sample_h, sample_w),
                        ).long(),
                        torch.zeros((sample_amt - 1, sample_h, sample_w)).long(),
                    ),
                ]
            ),
        },
        batch_size=[sample_amt, sample_h, sample_w],
    )
    storage[PRED_PANOPTIC]
    storage.memmap_()

    evaluator = PanopticEvaluator.from_metadata(
        "cityscapes", show_progress=True, show_summary=True, show_details=True
    )
    metrics = evaluator.compute(storage, device="cpu")

    print(metrics)
