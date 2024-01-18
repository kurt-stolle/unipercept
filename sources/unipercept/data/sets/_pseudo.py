from __future__ import annotations

import functools

import safetensors.torch as safetensors
import torch
import torch.utils.data
from PIL import Image

from unipercept import file_io

__all__ = ["PseudoGenerator"]


class PseudoGenerator:
    def __init__(self, depth_model="sayakpaul/glpn-kitti-finetuned-diode-221214-123047", depth_factor: float = 8.0):
        self.depth_name = depth_model
        self.depth_factor = depth_factor

    @functools.cached_property
    def depth_pipeline(self):
        from transformers import pipeline

        return pipeline(task="depth-estimation", model=self.depth_name, device="cuda")

    def create_panoptic_source(self, in_paths: tuple[file_io.Path, file_io.Path], out_path: file_io.Path):
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

    @torch.inference_mode()
    def _generate_depth(self, image: Image.Image) -> torch.Tensor:
        outputs = self.depth_pipeline(image)

        # interpolate to original size
        prediction = outputs["predicted_depth"]
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).cpu()
        return prediction.cpu() * self.depth_factor

    def create_depth_source(self, img_path: file_io.Path | str, out_path: file_io.Path | str) -> None:
        """
        Uses pretrained DPT model to generate depth map pseudolabels
        """

        img_path = file_io.Path(img_path)
        out_path = file_io.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert out_path.name.endswith(".safetensors"), f"Expected {out_path} to end with .safetensors!"

        image = Image.open(img_path)
        for output in self._generate_depth(image):
            safetensors.save_file(
                {"data": output.as_subclass(torch.Tensor)}, out_path, metadata={"model": self.depth_name}
            )

    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.v2.functional import to_pil_image

        image_pil = to_pil_image(image)
        return self.depth_pipeline(image_pil)["predicted_depth"]  # type: ignore
