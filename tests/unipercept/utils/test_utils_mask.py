from __future__ import annotations

import pytest
import torch

center_cases = [
    # Fully occupied mask, center in middle
    (torch.full((1, 5, 3), True, dtype=torch.bool), [(1, 2)]),
    # Middle pixel only, center in middle
    (
        torch.tensor(
            [[i == 1 and j == 1 for i in range(3)] for j in range(3)], dtype=torch.bool
        ).unsqueeze_(0),
        [(1, 1)],
    ),
    # Left row only, pixel left-middle
    (torch.arange(3).unsqueeze(0).repeat(3, 1).unsqueeze_(0) < 1, [(0, 1)]),
    # Unbalanced center
    (torch.arange(3 * 3).reshape(1, 3, 3).bool(), [(1.125, 1.125)]),
]


@pytest.mark.unit()
@pytest.mark.parametrize("i,o", center_cases)
def test_mask_to_centers(i: torch.Tensor, o: torch.Tensor):
    from unipercept.utils.mask import masks_to_centers

    r = masks_to_centers(i, stride=1, use_vmap=True)

    assert r.dtype == torch.float

    o_t = torch.as_tensor(o, dtype=r.dtype)

    assert r.shape[-1] == 2
    assert torch.allclose(r, o_t)

    r_vmap = masks_to_centers(i, stride=1, use_vmap=False)
    assert torch.allclose(r_vmap, r), "Vmap and non-vmap should be equal"
    assert torch.allclose(r_vmap, o_t)


box_cases = [
    # Fully occupied mask, center in middle
    (torch.full((5, 3), True, dtype=torch.bool), (0, 0, 2, 4)),
    # Middle pixel only, center in middle
    (
        torch.tensor(
            [[i == 1 and j == 1 for i in range(3)] for j in range(3)], dtype=torch.bool
        ),
        (1, 1, 1, 1),
    ),
    # Left row only, pixel left-middle
    (torch.arange(3).unsqueeze(0).repeat(3, 1) < 1, (0, 0, 0, 2)),
    # Unbalanced center
    (torch.arange(3 * 3).reshape(3, 3).bool(), (0, 0, 2, 2)),
]


@pytest.mark.unit()
@pytest.mark.parametrize("i,o", box_cases)
def test_mask_to_boxes(i: torch.Tensor, o: tuple[float, float]):
    from unipercept.utils.mask import masks_to_boxes

    i = i.unsqueeze(0).repeat(2, 1, 1)
    r = masks_to_boxes(i)
    print(r)

    assert r.dtype == torch.float

    o_t = torch.as_tensor(o, dtype=r.dtype).unsqueeze(0).repeat(2, 1)
    assert r.shape[-1] == 4
    assert torch.allclose(r, o_t)


@pytest.mark.unit()
def test_masks_to_boxes_batch():
    from unipercept.utils.mask import masks_to_boxes

    i = torch.nn.utils.rnn.pad_sequence([c[0] for c in box_cases], batch_first=True)
    r = masks_to_boxes(i)
    print(r)
    assert r.dtype == torch.float

    o_t = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(c[1], dtype=r.dtype) for c in box_cases], batch_first=True
    )

    assert r.shape[-1] == 4
    assert torch.allclose(r, o_t)
