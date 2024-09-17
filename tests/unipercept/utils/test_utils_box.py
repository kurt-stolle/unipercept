from __future__ import annotations

import itertools

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torchvision.tv_tensors import BoundingBoxFormat as TVBBoxFormat
from unipercept.utils.box import BBoxFormat, convert_boxes

# Define a strategy for generating boxes
boxes_strategy = st.lists(
    st.floats(allow_nan=False, allow_infinity=False), min_size=4, max_size=4
)

# Defines all combinations of box formats to test
conversion_strategy = st.sampled_from(
    list(
        itertools.product(list(iter(BBoxFormat)) + list(iter(TVBBoxFormat))),
    )
)


@given(boxes=boxes_strategy)
@given(conversion=conversion_strategy)
def test_convert_boxes(boxes, conversion):
    r"""
    Tests conversions fron and to different bounding box formats
    """
    boxes_tensor = torch.Tensor(boxes)
    src_format, dst_format = conversion

    # Some cases may raise NotImplementedError - pytest should xfail in this case

    try:
        result = convert_boxes(boxes_tensor, src_format, dst_format)
    except NotImplementedError:
        pytest.xfail("Conversion not implemented")
    assert isinstance(result, torch.Tensor)

    try:
        result = convert_boxes(result, dst_format, src_format)
    except NotImplementedError:
        pytest.xfail("Conversion not implemented")
    assert isinstance(result, torch.Tensor)
