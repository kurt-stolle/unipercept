# UniPercept

## Installation

This package requires at least Python 3.11 and PyTorch 2.1. Once you have created an environment with these 
dependencies, we can proceed to install `unipercept` using one of three installation methods.

### Stable release (recommended)
You can install the latest stable release from PyPI via
```bash
pip install unipercept
```

### Master branch
To install the latest version, which is not guaranteed to be stable, install from GitHub using 
```bash
pip install git+https://github.com/kurt-stolle/unipercept.git
```

### Developers
If your use-case requires changes to our codebase, we recommend that you first fork this repository and download your
own fork locally. Assuming you have the GitHub CLI installed, you can clone your fork with
```bash
gh repo clone unipercept
```
Then, you can proceed to install the package in editable mode by running
```bash
pip install --editable unipercept
```
You are invited to share your improvements to the codebase through a pull request on this repository. 
Before making a pull request, please ensure your changes follow our code guidelines by running `pre-commit` before 
adding your files into a Git commit.

## Training and evaluation

Models can be trained and evalurated from the CLI or through the Python API. 

### CLI
To train a model with the CLI:
```bash
unicli train --config <config path>
```
Without a `<config name>`, an interactive prompt will be started to assist in finding a configuration file.

## Developer guidelines
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
