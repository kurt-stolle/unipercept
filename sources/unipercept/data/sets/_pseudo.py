from __future__ import annotations

import functools
from pathlib import Path

import safetensors.torch as safetensors
import torch
import torch.nn as nn

__all__ = ["PseudoGenerator"]


class PseudoGenerator:
    def __init__(self, depth_model="sayakpaul/glpn-kitti-finetuned-diode-221214-123047"):
        self.depth_name = depth_model

    @functools.cached_property
    def depth_pipeline(self):
        # from transformers import DPTForDepthEstimation, DPTImageProcessor

        # processor = DPTImageProcessor.from_pretrained(self.depth_name)  # type: ignore
        # model: nn.Module = DPTForDepthEstimation.from_pretrained(self.depth_name)  # type: ignore
        # model.eval()

        # return processor, model
        from transformers import pipeline

        return pipeline(task="depth-estimation", model=self.depth_name, device="cuda")

    def create_panoptic_source(self, in_paths: tuple[Path, Path], out_path: Path):
        import numpy as np
        from PIL import Image

        from unipercept.data.tensors import PanopticMap

        pan_path = out_path
        seg_path, ins_path = in_paths

        assert seg_path.is_file(), f"Expected {seg_path} to exist!"
        assert ins_path.is_file(), f"Expected {ins_path} to exist!"
        assert not pan_path.is_file(), f"Expected {pan_path} to not exist!"

        seg = np.asarray(Image.open(seg_path))
        ins = np.asarray(Image.open(ins_path))
        assert seg.size == ins.size, f"Expected same size, got {seg.size} and {ins.size}!"
        assert seg.ndim == ins.ndim == 2, f"Expected 2D images, got {seg.ndim}D and {ins.ndim}D!"

        pan = PanopticMap.from_parts(semantic=seg, instance=ins).cpu()

        pan_path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.save_file({"data": pan.as_subclass(torch.Tensor)}, pan_path)

    def create_depth_source(self, img_path: Path, out_path: Path) -> None:
        """
        Uses pretrained DPT model to generate depth map pseudolabels
        """
        from PIL import Image

        image = Image.open(img_path)
        # processor, model = self.depth_pipeline

        # # prepare image for the model
        # inputs = processor(images=image, return_tensors="pt")  # type: ignore

        with torch.inference_mode():
            # outputs = model(**inputs)  # type: ignore
            # predicted_depth = outputs.predicted_depth

            outputs = self.depth_pipeline(image)

            # interpolate to original size
            prediction = outputs["predicted_depth"]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).cpu()

        assert out_path.name.endswith(".safetensors"), f"Expected {out_path} to end with .safetensors!"

        output = prediction.squeeze(0).cpu()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.save_file({"data": output.as_subclass(torch.Tensor)}, out_path, metadata={"model": self.depth_name})

    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.v2.functional import to_pil_image

        image_pil = to_pil_image(image)
        return self.depth_pipeline(image_pil)["predicted_depth"]  # type: ignore
