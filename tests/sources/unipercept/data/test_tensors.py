from unipercept.data import tensors


def test_pixel_map_registry():
    reg = tensors.registry.pixel_maps
    assert reg is not None

    known_pixel_maps = (tensors.OpticalFlow, tensors.Mask, tensors.Image, tensors.PanopticMap)

    assert len(reg) >= len(known_pixel_maps)

    for t in known_pixel_maps:
        assert t in reg, f"{t} not registered in {reg}!"
