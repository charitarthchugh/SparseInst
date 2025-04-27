FROM nvcr.io/nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV TZ=US/Eastern
ENV DETECTRON_TAG=v0.6
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update && apt install build-essential ninja-build vim git g++ ffmpeg libsm6 libxext6 python3.11-full python3.11-dev -y
RUN python3.11 -m ensurepip
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    python3.11 -m pip install opencv-python opencv-contrib-python scipy wandb tensorboard

WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2/ && git checkout tags/${DETECTRON_TAG} && python3.11 setup.py build develop

RUN  python3.11 -m pip install iopath fvcore portalocker yacs timm pyyaml==5.1 shapely requests

RUN git clone https://github.com/hustvl/SparseInst
WORKDIR /workspace/SparseInst
RUN ln -s /usr/bin/python3.11 /usr/bin/python
ENTRYPOINT bash
