from unipercept.data import points as data_points


def test_pixel_map_registry():
    reg = data_points.registry.pixel_maps
    assert reg is not None

    known_pixel_maps = (data_points.OpticalFlow, data_points.Mask, data_points.Image, data_points.PanopticMap)

    assert len(reg) >= len(known_pixel_maps)

    for t in known_pixel_maps:
        assert t in reg, f"{t} not registered in {reg}!"
