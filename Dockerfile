FROM nvidia/cuda:12.4.1-cudnn-devel-ubi9

# Set FORCE_CUDA because during `docker build` cuda is not accessible. Provide
# a build argument for TORCH_CUDA_ARCH_LIST to specify the compute capabilities,
# which should increase build speed.
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV CUDA_HOME="/usr/local/cuda-12.4/"

# Install global dependencies
# NOTE: Previous versions used updated packages as  below, which could result in 
#       stability issues
#RUN dnf update -y && \
#    dnf install -y sudo wget make gcc bzip2-devel openssl-devel zlib-devel libffi-devel ninja-build
RUN dnf install -y  \
    sudo wget \ 
    make gcc ninja-build \
    bzip2-devel openssl-devel zlib-devel libffi-devel ncurses-devel xz-devel \
    python3.12 python3.12-pip python3.12-devel python3.12-libs python3.12-wheel \
    mesa-libGL 

# Install Python
#ARG PYTHON_VERSION_MAJOR=3.12
#ARG PYTHON_VERSION_MINOR=6
#RUN mkdir /tmp/python && \
#    cd /tmp/python && \
#    wget https://www.python.org/ftp/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.tgz && \
#    tar -xzf Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.tgz && \
#    cd Python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} && \
#    ./configure --enable-optimizations && \
#    make -j$(nproc) && \
#    make altinstall

# Create a non-root user named 'perciever'.
# Build with --build-arg USER_ID=$(id -u) to use your own user id.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} perciever
RUN usermod -aG wheel perciever
RUN echo '%wheel ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER perciever
WORKDIR /home/perciever
RUN mkdir -p /home/perciever/.local/bin
ENV PATH="/home/perciever/.local/bin:$PATH"

# Create virtual environment
#RUN python${PYTHON_VERSION_MAJOR} -m venv --system-site-packages /home/perciever/.venv
#ENV PATH="/home/perciever/.venv/bin:$PATH"
#ENV PYTHON="/home/perciever/.venv/bin/python"
#ENV PIP="/home/perciever/.venv/bin/pip"

ENV PYTHON="/usr/bin/python3.12"
ENV PIP="/usr/bin/pip3.12"


# Install PyTorch before installing other dependencies
RUN $PIP install nvidia-pyindex 
RUN $PIP install nvidia-nccl
RUN $PIP install torch torchvision  --index-url https://download.pytorch.org/whl/cu124

# Set a fixed model cache directory for FVCORE
RUN sudo mkdir -p /tmp/fvcore_cache && sudo chown perciever:perciever /tmp/fvcore_cache
ENV FVCORE_CACHE="/tmp/fvcore_cache"

# Install python package
RUN mkdir -p /home/perciever/unipercept
WORKDIR /home/perciever/unipercept
ADD . .
RUN sudo chown -R perciever:perciever .
RUN $PIP install .

# Output, configs and models directory are not added (see Dockerignore) and
# should be mounted as volumes. We create an empty target directory for each.

# Configure output and datasets directories
RUN sudo mkdir -p /srv/data /srv/output && sudo chown perciever:perciever /srv/data /srv/output
ENV UP_DATASETS=/srv/data
ENV UP_OUTPUT=/srv/output

ENTRYPOINT ["python3.12", "-m", "unipercept"]
