# Unified Perception: Efficient Video Panoptic Segmentation with Minimal Annotation Costs

Welcome to the PyTorch 2 implementation of Unified Perception, an innovative approach to depth-aware video panoptic segmentation. Our method achieves state-of-the-art performance without the need for video-based training. Instead, it utilizes a two-stage cascaded tracking algorithm that reuses object embeddings computed in an image-based network. This repository mirrors the research paper [Unified Perception](https://arxiv.org/abs/2303.01991).

## Introduction

Unified Perception is designed to tackle the challenge of depth-aware video panoptic segmentation with high precision. The method proposed in this repository effectively combines the benefits of image-based networks with a two-stage cascaded tracking algorithm, ultimately enhancing the performance of video panoptic segmentation tasks without the need for video-based training.

## Technical Documentation

For a comprehensive understanding of the Unified Perception implementation, please visit our technical documentation hosted on:

[https://tue-mps.github.io/unipercept](https://tue-mps.github.io/unipercept)

## Usage & Installation (Coming Soon)

_This section is under development and will be added in the near future. It will provide step-by-step instructions for using and installing the Unified Perception package. Consider starring or subscribing to this repository for updates._

The following _optional_ packages may be installed:
```bash
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

## Training and evaluation

Use the CLI:
```bash
unicli train --config <config name>
```

Without a `<config name>`, an interactive prompt will be started.

## Validation and testing
All tests can ran via `python -m pytest`. 
However, we also provide a `make` directive that uses `pytorch-xdist` to speed up the process:
```
make test
```
You may need to tune the parameters if memory problems arise during testing. 

Similarly, benchmarks are implemented using `pytest-benchmark`. To run them, use:
```
make benchmark
```

Coverage reports are built using `pytest-cov`, and executed using:
```
make coverage
```

Last, we largely follow the same principles and methods as [Transformers](https://huggingface.co/docs/transformers) uses for testing. 
For more information on using `pytest` for automated testing, check out [their documentation](https://huggingface.co/transformers/v3.4.0/testing.html).

## Acknowledgements

We would like to express our gratitude to the developers of the following open-source projects, which have significantly contributed to the success of our work:

- [PyTorch](https://github.com/pytorch/pytorch): An open-source machine learning framework that accelerates the path from research prototyping to production deployment.
- [Detectron2](https://github.com/facebookresearch/detectron2): A platform for object detection and segmentation built on PyTorch. We liberally use the packages and code from this project.
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d): A library on which we base our camera projection from 2D to 3D space.
- [Panoptic FCN](https://github.com/DdeGeus/PanopticFCN_cityscapes): An implementation of the Panoptic FCN method for panoptic segmentation tasks.
- [ViP-DeepLab](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/vip_deeplab.md): The baseline implementation for the depth-aware video panoptic segmentation task.
- [Panoptic Depth](https://github.com/NaiyuGao/PanopticDepth): A repository that implements the instance (de)normalization procedure that significantly improves depth estimation for _things_.

The Unified Perception implementation contains extracts of the above repositories that have been edited to suit the specific needs of this project.
Whenever possible, the original libraries are used instead.

## License

This repository is released under the MIT License. For more information, please refer to the LICENSE file.
