#!/bin/bash
#
set -e

ENV_NAME="unipercept"
EXTRA="tests,docs"
PYTHON="conda run -n $ENV_NAME python"

# Create conda environment
    # name: unipercept
    # channels:
    #   - nvidia/label/cuda-12.1.1
    #   - defaults
    # dependencies:
    #   - python=3.11
    #   - cuda
    #   - cuda-toolkit
# conda env create --file conda.yml
conda create -n "$ENV_NAME" -y python=3.11 cuda cuda-toolkit -c nvidia/label/cuda-12.1.1

# Install PyTorch
$PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies only available via Git
$PYTHON -m pip install \
    'git+https://github.com/facebookresearch/detectron2.git' \
    'git+https://github.com/autonomousvision/kitti360Scripts.git'

# Install alternative LAP package
$PYTHON -m pip install lapx

# Install UniPercept
$PYTHON -m pip install -e ".[$EXTRA]"

# Install Pillow-SIMD
$PYTHON -m pip uninstall -y pillow
CC="cc -mavx2" $PYTHON -m pip install --force-reinstall pillow-simd

# Done!
exit
