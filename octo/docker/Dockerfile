FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig
RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    software-properties-common \
    python3

RUN apt-get update && \
    apt-get install -y \
    python3-pip

# === Project-specific stuff ===
RUN pip3 install torch einops numpy

RUN pip3 install \
    wandb \
    kornia \
    imageio \
    imageio-ffmpeg \
    moviepy

RUN pip3 install pydantic

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget

ADD requirements.txt .
RUN pip3 install -r requirements.txt

# Set up for Xdummy
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
WORKDIR /opt
COPY ./vendor ./vendor
COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy
RUN apt-get update && apt-get install -y --no-install-recommends \
    xserver-xorg-video-dummy

RUN pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Install ACT repository
RUN git clone https://github.com/tonyzhaozh/act.git
# RUN cd act/detr && pip install -e .
RUN pip3 install \
    opencv-python \
    modern_robotics \
    pyrealsense2 \
    h5py_cache \
    pyquaternion \
    pyyaml \
    rospkg \
    pexpect \
    mujoco==2.3.3 \
    dm_control==1.0.9 \
    einops \
    packaging \
    h5py \
    ipython


# Insatll language-table env
# RUN git clone https://github.com/google-research/language-table.git
# RUN pip3 install tf-nightly==2.16.0.dev20231109
# RUN cd language-table && \
#     sed -i 's/tf-nightly>=2.12.0.dev20230201/tf-nightly==2.16.0.dev20240205/' requirements.txt && \
#     pip install -r requirements.txt

ENTRYPOINT ["/opt/vendor/Xdummy-entrypoint"]
