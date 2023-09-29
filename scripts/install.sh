#!/bin/bash



PYTHON=python3
PIP=python3 -m pip

E_PYTORCH_NOT_FOUND=1

echo "Installing UniPercept..."

# Output Python location
echo "Python location: $($PYTHON -c "import sys; print(sys.executable)")"

# Check if PyTorch is installed
if ! $PYTHON -c "import torch" &> /dev/null; then
    echo "PyTorch not found. Installing PyTorch first."
    exit E_PYTORCH_NOT_FOUND
fi

# Install UniPercept
$PIP install -e -y .

echo "UniPercept installed successfully."

# Install optional Pillow-SIMD
echo "Installing Pillow-SIMD..."

$PIP uninstall -y pillow
CC="cc -mavx2" $PIP install --force-reinstall pillow-simd

echo "Pillow-SIMD installed successfully."

exit