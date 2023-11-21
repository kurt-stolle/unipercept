# This file must be sourced from the shell, not ran directly.
# It initializes the environment for running UniPercept on the Snellius HPC cluster provided by SURF.

GCC_VERSION="12.3.0"
GCC_PACKAGE="GCCcore-${GCC_VERSION}"

CUDA_VERSION="12.1.1"
CUDA_PACKAGE="CUDA-${CUDA_VERSION}"

# Load base module - 2023 currently in production
module purge
module load 2023

# Python, CUDA, and other libraries
module load Python/3.11.3-${GCC_PACKAGE}
module load CUDA/${CUDA_VERSION}
module load NVHPC/23.7-${CUDA_PACKAGE}
module load NCCL/2.18.3-${GCC_PACKAGE}-${CUDA_PACKAGE}
module load cuDNN/8.9.2.26-${CUDA_PACKAGE}
module load HDF5/1.14.0-gompi-2023a
module load ImageMagick/7.1.1-15-${GCC_PACKAGE}
module load h5py/3.9.0-foss-2023a
module load dill/0.3.7-${GCC_PACKAGE}
module load FFTW.MPI/3.3.10-gompi-2023a
module load ScaLAPACK/2.2.0-gompi-2023a-fb
module load flatbuffers-python/23.5.26-${GCC_PACKAGE}
module load protobuf-python/4.24.0-${GCC_PACKAGE}
module load SciPy-bundle/2023.07-gfbf-2023a
module load pybind11/2.11.1-${GCC_PACKAGE}
module load PyYAML/6.0-${GCC_PACKAGE}
module load FFmpeg/6.0-${GCC_PACKAGE}
module load Pillow/10.0.0-${GCC_PACKAGE}
