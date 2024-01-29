from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX


@pytest.mark.parametrize("images_subfolder", ["samples-static", "samples-video"])
def test_prepare_images(images_subfolder):
    from unipercept import prepare_images

    images, info = prepare_images(
        "./configs/cityscapes/multidvps_resnet50.py", f"./assets/{images_subfolder}"
    )
    sample = next(images)

    print(sample)
    print(info)
