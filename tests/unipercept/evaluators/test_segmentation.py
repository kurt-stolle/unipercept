from __future__ import annotations

import pprint

import pytest
import torch
import unipercept.evaluators.segmentation as segeval
from tensordict import TensorDict


def test_panoptic_evaluator():
    """
    Test the panoptic evaluator.
    """

    from unipercept.data.tensors import Image, PanopticMap
    from unipercept.model import CaptureData, InputData

    sample_w = 32
    sample_h = 16
    true_panoptic = PanopticMap.from_parts(
        torch.randint(0, 20, (sample_h, sample_w)).long(),
        torch.randint(0, 20, (sample_h, sample_w)).long(),
    )
    pred_panoptic = PanopticMap.from_parts(
        torch.randint(0, 20, (sample_h, sample_w)).long(),
        torch.randint(0, 20, (sample_h, sample_w)).long(),
    )
    evaluator = segeval.PanopticSegmentationEvaluator.from_metadata(
        "cityscapes", show_progress=True, show_summary=True, show_details=True
    )

    storage_parts = []
    for i, (true_pan, pred_sem) in enumerate(
        [
            (true_panoptic, true_panoptic),
            (true_panoptic, pred_panoptic),
            (true_panoptic, torch.zeros_like(pred_panoptic)),
        ]
    ):
        storage = TensorDict({}, batch_size=[])
        inputs = InputData(
            ids=torch.tensor([i, 0]),
            captures=CaptureData(
                images=torch.zeros(1, 3, sample_h, sample_w).as_subclass(Image),
                segmentations=true_pan.unsqueeze(0).as_subclass(PanopticMap),
                times=torch.zeros(1),
                batch_size=[1],
            ),
            batch_size=[],
        ).unsqueeze(0)
        predictions = {"panoptic_segmentation": pred_sem.unsqueeze(0)}
        evaluator.update(storage, inputs, predictions)
        storage_parts.append(storage)

    storage_all = torch.stack(storage_parts)
    storage_all.memmap_()

    metrics = evaluator.compute(storage_all, device="cpu", path=".")
    pprint.pprint(metrics)


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

    for a, b in zip(
        (z_pc, z_all), segeval.compute_semantic_miou(a, b, n), strict=False
    ):
        assert torch.allclose(a, b), (a.tolist(), b.tolist())


def test_semantic_evaluator():
    """
    Test the panoptic evaluator.
    """
    from unipercept.data.tensors import Image, PanopticMap
    from unipercept.model import CaptureData, InputData

    sample_amt = 3
    sample_w = 32
    sample_h = 16
    true_panoptic = PanopticMap.from_parts(
        torch.randint(0, 20, (sample_h, sample_w)).long(),
        torch.zeros(sample_h, sample_w).long(),
    )
    pred_semantic = torch.randint(0, 20, (sample_h, sample_w)).long()
    evaluator = segeval.SemanticSegmentationEvaluator.from_metadata(
        "cityscapes", show_progress=True, show_summary=True, show_details=True
    )

    storage_parts = []
    for i, (true_pan, pred_sem) in enumerate(
        [
            (true_panoptic, true_panoptic),
            (true_panoptic, pred_semantic),
            (true_panoptic, torch.zeros_like(pred_semantic)),
        ]
    ):
        storage = TensorDict({}, batch_size=[])
        inputs = InputData(
            ids=torch.tensor([i, 0]),
            captures=CaptureData(
                images=torch.zeros(1, 3, sample_h, sample_w).as_subclass(Image),
                segmentations=true_pan.unsqueeze(0).as_subclass(PanopticMap),
                times=torch.zeros(1),
                batch_size=[1],
            ),
            batch_size=[],
        ).unsqueeze(0)
        predictions = {"semantic_segmentation": pred_sem.unsqueeze(0)}
        evaluator.update(storage, inputs, predictions)
        storage_parts.append(storage)

    storage_all = torch.stack(storage_parts)
    storage_all.memmap_()

    metrics = evaluator.compute(storage_all, device="cpu", path=".")
    pprint.pprint(metrics)
