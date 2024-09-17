#!/usr/bin/env python
"""
These are some notes on how to render a 3D pointcloud with Open3d.
"""

from __future__ import annotations

import random
import time
import typing as T

import open3d as o3d
import torch
import torchvision.transforms.v2.functional as tvfn
import unipercept as up

MAX_POINTS: int | None = None  # int(1e5)

RENDER_BATCH_SIZE: int = int(1e3)
RENDER_DELTA = 1 / 120  # 1/FPS

DRAW_FLOOR = False
DRAW_ORIGIN = True

random.seed(0)


# Get the render option and set up the camera
def camera_to_o3d(cam):
    cam_o3d = o3d.camera.PinholeCameraParameters()
    cam_o3d.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, cam.K[0, :, :])
    cam_o3d.extrinsic = cam.E[0, :, :]

    return cam_o3d


def take_samples(points, *other):
    idx_queue = list(range(points.shape[0]))
    idx_queue = random.sample(idx_queue, min(len(idx_queue), RENDER_BATCH_SIZE))

    res = tuple(t[idx_queue] for t in (points, *other))

    keep = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    keep[idx_queue] = False

    queue = tuple(t[keep] for t in (points, *other))

    return res, queue


if __name__ == "__main__":
    # Sample a depthmap and pointcloud from the dataset
    up.log.logger.info("Loading dataset...")
    dataset = up.data.sets.catalog.get_dataset("cityscapes-vps")(split="val", all=False)
    queue, pipe = dataset()
    sample = T.cast(up.model.InputData, next(iter(pipe)))

    # Depthmap, image and camera params
    dmap = sample.captures.depths
    image = sample.captures.images
    cam = sample.cameras

    # Render the image
    up.log.logger.info("Rendering image...")
    render = up.render.draw_image(image)
    up.render.terminal.show(render)

    # Apply projection
    up.log.logger.info("Projecting image...")
    dmap_3d = cam.reproject_map(dmap)
    points = dmap_3d[dmap > 0]
    points = up.vision.geometry.convert_points(points, tgt="open3d")
    colors = tvfn.pil_to_tensor(render).permute(1, 2, 0).unsqueeze(0)[dmap > 0] / 255

    if MAX_POINTS:
        idx = random.sample(range(points.shape[0]), min(MAX_POINTS, points.shape[0]))
        points = points[idx]
        colors = colors[idx]

    queue = (points, colors)

    # Create Open3D PointCloud object
    up.log.logger.info("Setting up Open3D scene...")

    # Set up a renderer
    height, width = list(map(int, cam.canvas_size[0].tolist()))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.get_render_option().background_color = [0.2, 0.2, 0.2]

    # Add floor plane
    if DRAW_FLOOR:
        up.log.logger.info("Drawing floor plane...")

        floor_size = 100
        floor = o3d.geometry.TriangleMesh.create_box(
            width=floor_size, height=0.01, depth=floor_size
        )
        floor.translate([-floor_size / 2, 0.0, -floor_size / 2])
        floor.paint_uniform_color([0.7, 0.7, 0.7])

        vis.add_geometry(floor)

    # Add cooridnate frame
    if DRAW_ORIGIN:
        up.log.logger.info("Drawing coordinate frame... (X=red, Y=green, Z=blue)")

        cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5, origin=[0, 0, 0]
        )
        vis.add_geometry(cframe)

    # Add pointcloud
    (points, colors), queue = take_samples(*queue)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)  # colors.numpy())

    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    # ctr.set_front([1, 0, 0])
    # ctr.set_up([0, 0, 1])
    ctr.set_lookat((0, 0, 1))
    # ctr.set_lookat(pcd.get_center())

    # Start the visualization
    t_render: float | None = time.time()
    visualize = True
    up.log.logger.info("Launching visualization... Press [H] to view controls.")
    while visualize:
        if t_render is not None and time.time() > t_render:
            (points, colors), queue = take_samples(*queue)
            n_queued = len(queue[0])

            pcd.points.extend(points)
            pcd.colors.extend(colors)
            vis.update_geometry(pcd)
            t_render = (time.time() + RENDER_DELTA) if n_queued > 0 else None

        visualize = vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()

    up.log.logger.info("Done!")
