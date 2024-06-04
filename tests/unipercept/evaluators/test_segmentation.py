from __future__ import annotations

import typing as T

import einops
import pytest
import torch
import typing_extensions as TX
from tensordict import TensorDict

import unipercept.evaluators.segmentation as segeval


def test_panoptic_evaluator(
    true_panoptic: torch.Tensor, pred_panoptic: torch.Tensor, true_info
):
    """
    Test the panoptic evaluator.
    """

    from unipercept.data.tensors import PanopticMap

    sample_h, sample_w = true_panoptic.shape
    sample_amt = 3

    print(f"GT: {true_panoptic.unique().tolist()}")

    evaluator = segeval.PanopticSegmentationEvaluator.from_metadata(
        "cityscapes",
        show_progress=True,
        show_summary=True,
        show_details=True,
        segmentation_task=segeval.SegmentationTask.PANOPTIC_SEGMENTATION,
    )

    storage = TensorDict(
        {
            "true_panoptic_segmentation": einops.repeat(
                true_panoptic, "h w -> b h w", b=sample_amt
            ),
            "pred_panoptic_segmentation": torch.cat(
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
    storage.memmap_()

    evaluator = segeval.PanopticSegmentationEvaluator.from_metadata(
        "cityscapes", show_progress=True, show_summary=True, show_details=True
    )
    metrics = evaluator.compute(storage, device="cpu", path=".")

    print(metrics)


@pytest.mark.parametrize(
    "a,b,n,y_pc,y_all",
    [
        ([[0, 1]], [[0, 1]], 2, [1.0, 1.0], 1.00),
        ([[0, 1]], [[1, 1]], 2, [0.0, 0.5], 0.25),
        ([[0, 1]], [[0, 0]], 2, [0.5, 0.0], 0.25),
        ([[0, 0]], [[1, 1]], 2, [0.0, 0.0], 0.00),
        ([[0, 1]], [[1, 0]], 2, [0.0, 0.0], 0.00),
        ([[0, 1]], [[1, 0]], 3, [0.0, 0.0, 0.0], 0.00),
        ([[0, 1]], [[1, 2]], 3, [0.0, 0.0, 0.0], 0.00),
        ([[0, 1, 2]], [[1, 2, 2]], 3, [0.0, 0.0, 0.5], 1 / 6),
    ],
)
def test_segmentation_miou(a, b, n, y_pc, y_all):
    from unipercept.data.tensors import PanopticMap

    a = torch.tensor(a)
    b = torch.tensor(b)
    a, b = (PanopticMap.from_parts(t, torch.zeros_like(t)) for t in (a, b))
    inter, union = segeval.compute_semantic_miou_partial(a, b, n)
    z_pc, z_all = segeval.accumulate_semantic_miou_partial(inter, union)
    classes = list(range(n))
    show_dict = {
        "a_sem": a.get_semantic_map().tolist(),
        "b_sem": b.get_semantic_map().tolist(),
        "inter": dict(zip(classes, inter.tolist(), strict=True)),
        "union": dict(zip(classes, union.tolist(), strict=True)),
        "miou_per_class": z_pc.tolist(),
        "miou_all": z_all.tolist(),
    }
    print(
        "Computed mean IoU: \n\t"
        + "\n\t".join(f"{k}: {v}" for k, v in show_dict.items())
    )

    y_pc = torch.tensor(y_pc)
    assert torch.allclose(z_pc, y_pc), (z_pc.tolist(), y_pc.tolist())
    y_all = torch.tensor(y_all)
    assert torch.allclose(z_all, y_all), (z_all.tolist(), y_all.tolist())

    for a, b in zip((z_pc, z_all), segeval.compute_semantic_miou(a, b, n)):
        assert torch.allclose(a, b), (a.tolist(), b.tolist())
