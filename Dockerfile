FROM nvidia/cuda:12.1.0-cudnn8-devel-ubi8

# Set FORCE_CUDA because during `docker build` cuda is not accessible. Provide
# a build argument for TORCH_CUDA_ARCH_LIST to specify the compute capabilities,
# which should increase build speed.
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install global dependencies
RUN dnf update -y && dnf install -y sudo wget make gcc bzip2-devel openssl-devel zlib-devel libffi-devel ninja-build

# Install Python
ARG PYTHON_VERSION_MAJOR=3.11
ARG PYTHON_VERSION_MINOR=7
RUN mkdir /tmp/python && \
    cd /tmp/python && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.tgz && \
    tar -xzf Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.tgz && \
    cd Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Create a non-root user named 'perciever'.
# Build with --build-arg USER_ID=$(id -u) to use your own user id.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} perciever -g sudo
RUN echo '%wheel ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER perciever
WORKDIR /home/perciever

# Create virtual environment
RUN python${PYTHON_VERSION_MAJOR} -m venv --system-site-packages /home/perciever/.venv
ENV PATH="/home/perciever/.venv/bin:$PATH"
ENV PYTHON="/home/perciever/.venv/bin/python"
ENV PIP="/home/perciever/.venv/bin/pip"

# Install PyTorch before installing other dependencies
RUN $PIP pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Set a fixed model cache directory for FVCORE
RUN sudo mkdir -p /tmp/fvcore_cache && sudo chown perciever:perciever /tmp/fvcore_cache
ENV FVCORE_CACHE="/tmp/fvcore_cache"

# Install Unified Perception python package
RUN mkdir -p /home/perciever/unipercept
WORKDIR /home/perciever/unipercept
ADD . .
RUN $PIP install .

# Output, configs and models directory are not added (see Dockerignore) and
# should be mounted as volumes. We create an empty target directory for each.
RUN mkdir models configs

# Configure output and datasets directories
RUN sudo mkdir -p /srv/data /srv/output && sudo chown perciever:perciever /srv/data /srv/output
ENV UNICORE_DATASETS=/srv/data
ENV UNICORE_OUPUT=/srv/output

ENTRYPOINT ["$PYTHON", "-m", "unicli"]
