import pytest
import torch
import einops
from tensordict import TensorDict

from unipercept.evaluators import PanopticEvaluator, TRUE_PANOPTIC, PRED_PANOPTIC


@pytest.fixture(scope="module")
def true_info():
    """
    The true panoptic segmentation.
    """

    from unipercept.data.sets import get_info

    return get_info("kitti/step")


@pytest.fixture(scope="module")
def true_panoptic(true_info):
    """
    The true panoptic segmentation.
    """

    from unipercept.data.tensors import PanopticMap, LabelsFormat
    from torch.nn.functional import interpolate

    panseg = PanopticMap.read("assets/sample-annotated/segmentation.png", true_info, format=LabelsFormat.KITTI)

    panseg.unsqueeze_(1)
    panseg = interpolate(panseg.float(), scale_factor=0.25, mode="nearest").long()
    panseg.squeeze_(1)

    return panseg


def test_panoptic_evaluator(true_panoptic: torch.Tensor, true_info):
    """
    Test the panoptic evaluator.
    """

    from unipercept.data.tensors import PanopticMap

    sample_h, sample_w = true_panoptic.shape
    sample_amt = 3

    print(f"GT: {true_panoptic.unique().tolist()}")

    pred_panoptic = torch.where(true_panoptic >= 0, true_panoptic, torch.randint(0, 10, true_panoptic.shape))

    storage = TensorDict(
        {
            TRUE_PANOPTIC: einops.repeat(true_panoptic, "h w -> b h w", b=sample_amt),
            PRED_PANOPTIC: torch.cat(
                [
                    pred_panoptic.unsqueeze(0),
                    PanopticMap.from_parts(
                        torch.randint(
                            0, max(true_info.stuff_ids | true_info.thing_ids), (sample_amt-1, sample_h, sample_w)
                        ).long(),
                        torch.zeros((sample_amt-1, sample_h, sample_w)).long(),
                    ),
                ]
            ),
        },
        batch_size=[sample_amt, sample_h, sample_w],
    )
    storage[PRED_PANOPTIC]
    storage.memmap_()

    evaluator = PanopticEvaluator.from_metadata("cityscapes", show_progress=True, show_summary=True, show_details=True)
    metrics = evaluator.compute(storage, device="cpu")

    print(metrics)
