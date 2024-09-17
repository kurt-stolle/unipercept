from __future__ import annotations

import pytest
from unipercept.data import InferenceSampler, TrainingSampler


def test_inference_has_no_epochs():
    with pytest.warns(UserWarning, match="Epoch"):
        InferenceSampler(8, process_count=4, process_index=1, epoch=0)

    s = InferenceSampler(8, process_count=4, process_index=1)
    assert s._epoch == 0
    with pytest.raises(ValueError):
        e = s.epoch
        pytest.fail(f"Expected ValueError, but got {e=}")


def test_inference_sampler():
    q = 8
    s = InferenceSampler(q, process_count=4, process_index=1)
    assert s.total_count == q
    assert s.sample_count == q // s.process_count
    assert len(s) == 2

    expected_indices = [
        2,
        3,
    ]  # i.e. 2 elements with an offset of 1 from [0, 1, 2, 3, 4, 5, 6, 7]
    actual_indices = list(iter(s))

    assert len(actual_indices) == len(
        expected_indices
    ), f"{actual_indices=} != {expected_indices=}"
    assert all(
        a == b for a, b in zip(actual_indices, expected_indices, strict=False)
    ), f"{actual_indices=} != {expected_indices=}"


def test_training_sampler():
    items = 16
    rep_f = 1
    rep_r = 0

    sm_list = [
        TrainingSampler(
            items,
            process_count=4,
            process_index=i,
            epoch=0,
            selected_round=rep_r,
            repeat_factor=rep_f,
        )
        for i in range(4)
    ]
    for sm_1 in sm_list:
        assert sm_1.queue_size == items
        assert sm_1.total_count == items * rep_f
        assert sm_1.sample_count == items * rep_f // sm_1.process_count

    idx_list = [list(iter(sm)) for sm in sm_list]

    # Check lists are same size
    assert all(len(idx_list[0]) == len(idx) for idx in idx_list)

    for p_idx, s_idx in enumerate(idx_list):
        print(f"{p_idx=}: {s_idx=}")

        if rep_f == 1:
            for s_other in idx_list[:p_idx] + idx_list[p_idx + 1 :]:
                assert all(i not in s_other for i in s_idx)
