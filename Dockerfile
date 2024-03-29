# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.7 \
    python3.7-venv \
    python3.7-distutils \
    python3-pip \
    python3-apt \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.7 specifically
RUN python3.7 -m pip install --no-cache-dir --upgrade pip

# Copy the entire project and install dependencies
# Note: Consider using .dockerignore to exclude files not needed for the build
COPY requirements.txt /app/
COPY . /app
RUN pip install -e .
RUN pip3 install -r requirements.txt  --ignore-installed
RUN pip3 uninstall transformers -y
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN pip3 install git+https://github.com/jordiclive/transformers.git@controlprefixes --ignore-installed
RUN pip3 install torchtext==0.8.0 torch==1.7.1

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=
