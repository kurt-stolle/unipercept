"""
Tests ``unipercept.evaluators.dvps`` against the reference implementation.

Results to the reference implementation were computed ahead-of-time using the
steps listed in at the implementation <https://github.com/joe-siyuan-qiao/ViP-DeepLab>.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import safetensors.torch
import torch
from unipercept.evaluators.dvps import DVPSEvaluator

# From the reference implementation on the first sequence in Cityscapes-DVPS (val)
# using predictions from a fully trained Multi-DVPS model.
# Truths and predictions are saved in /assets/testing/dvps/
REFERENCE_ABS_REL = 0.0984
REFERENCE_DVPQ = [
    # frame, depth threshold, pq_all, pq_thing, pq_stuff
    [1, 0.5, 62.9142, 37.3028, 69.8992],
    [1, 0.25, 54.7778, 36.3051, 59.8158],
    [1, 0.1, 32.0433, 17.5942, 35.9840],
    [2, 0.5, 61.1277, 36.3485, 67.8857],
    [2, 0.25, 52.6446, 35.2948, 57.3763],
    [2, 0.1, 29.1442, 16.6292, 32.5574],
    [3, 0.5, 61.1248, 33.5826, 68.6363],
    [3, 0.25, 50.2899, 32.7297, 55.0791],
    [3, 0.1, 27.2493, 16.9008, 30.0717],
    [4, 0.5, 62.4153, 32.5191, 70.5688],
    [4, 0.25, 53.0557, 31.6233, 58.9009],
    [4, 0.1, 26.8842, 16.1433, 29.8135],
]
REFERENCE_VPQ = [
    # frame, pq_all, pq_thing, pq_stuff
    [1, 63.8452, 37.6274, 70.9955],
    [2, 61.9488, 36.7134, 68.8312],
    [3, 61.9198, 33.6917, 69.6184],
    [4, 63.1945, 32.6459, 71.5259],
]


@pytest.fixture(scope="module")
def results() -> dict[str, float]:
    """
    Write depth and panoptic predictions to a temporary directory and return the path
    to that directory
    """

    from tensordict import TensorDict
    from unipercept.data import tensors
    from unipercept.data.sets import catalog
    from unipercept.model import CaptureData, InputData

    evaluator = DVPSEvaluator.from_metadata(name="cityscapes-vps")

    root = Path(__file__).parent.parent.parent.parent / "assets" / "testing" / "dvps"
    assert root.is_dir(), root.as_posix()

    info = catalog.get_info("cityscapes-vps")

    # with MemmapTensorDictWriter(tmp_path) as storage:
    data_list = []
    for idx, id in enumerate(
        map(lambda f: f.stem[:13], root.glob("true/*_capture.png"))
    ):
        storage = TensorDict({}, batch_size=[1])
        outputs = TensorDict(
            safetensors.torch.load_file(root / "pred" / f"{id}.safetensors"),
            batch_size=[],
        ).unsqueeze(0)

        inputs = InputData(
            ids=torch.as_tensor([0, idx], dtype=torch.int64),
            captures=CaptureData(
                images=tensors.Image.read(
                    root / "true" / f"{id}_capture.png",
                ),
                depths=tensors.DepthMap.read(
                    root / "true" / f"{id}_depth.png",
                    format=tensors.DepthFormat.DEPTH_INT16,
                ),
                segmentations=tensors.PanopticMap.read(
                    root / "true" / f"{id}_segmentation.png",
                    info,
                    format=tensors.LabelsFormat.CITYSCAPES_VPS,
                ),
                times=torch.as_tensor(idx, dtype=torch.float32),
                batch_size=[],
            ).unsqueeze(0),
            batch_size=[],
        ).unsqueeze(0)

        evaluator.update(storage, inputs, outputs)
        data_list.append(storage)

    assert len(data_list) > 0

    data = torch.cat(data_list)

    results = evaluator.compute(
        data,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        return_dataframe=True,
    )
    return results


def test_dvps_evaluator_no_depth(results):
    df = results["vpq"]
    df = df.loc[(df["metric"] == "PQ") & (df["definition"] == "original")]
    df = df.groupby(["window"])[["all", "thing", "stuff"]].mean()
    print(df)
    for window, ref_all, ref_thing, ref_stuff in REFERENCE_VPQ:
        res_all, res_thing, res_stuff = df.loc[window]

        print(
            window, (ref_all, res_all), (ref_thing, res_thing), (ref_stuff, res_stuff)
        )

        assert ref_all == pytest.approx(res_all, abs=1e1)
        assert ref_thing == pytest.approx(res_thing, abs=1e1)
        assert ref_stuff == pytest.approx(res_stuff, abs=1e1)


def test_dvps_evaluator_with_depth(results):
    df = results["dvpq"]
    df = df.loc[(df["metric"] == "PQ") & (df["definition"] == "original")]

    abs_rel = df["abs_rel"].mean()
    print(f"mean abs_rel: {abs_rel:.4f}")

    df = df.groupby(["window", "threshold"])[
        ["all", "thing", "stuff", "abs_rel"]
    ].mean()
    print(df)

    # Check mean abs_rel
    assert abs_rel == pytest.approx(REFERENCE_ABS_REL, abs=1e-2)

    # Check per window/threshold pair
    for window, threshold, ref_all, ref_thing, ref_stuff in REFERENCE_DVPQ:
        res_all, res_thing, res_stuff, *_ = df.loc[(window, threshold)]

        print(
            window,
            threshold,
            (ref_all, res_all),
            (ref_thing, res_thing),
            (ref_stuff, res_stuff),
        )

        assert ref_all == pytest.approx(res_all, abs=1e1)
        assert ref_thing == pytest.approx(res_thing, abs=1e1)
        assert ref_stuff == pytest.approx(res_stuff, abs=1e1)
