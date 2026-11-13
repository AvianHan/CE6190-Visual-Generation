#!/bin/bash
# Separate evaluation environment for the pinned VBench (fd18b3d). Generation env is untouched.
set -ex
/home/ecs-user/miniconda3/bin/conda create -y -p /home/ecs-user/ce6190-runtime/a100x8-20261110/py310 --override-channels -c conda-forge python=3.10
/home/ecs-user/ce6190-runtime/a100x8-20261110/py310/bin/python -m venv /home/ecs-user/ce6190-runtime/a100x8-20261110/venv_vbench
P=/home/ecs-user/ce6190-runtime/a100x8-20261110/venv_vbench/bin/pip
$P install --upgrade pip wheel setuptools
$P install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$P install 'numpy<2.0.0' Pillow matplotlib 'timm>=0.9,<=1.0.12' cython tensorboard scipy opencv-python scikit-learn scikit-image openai-clip decord requests pyyaml easydict pyiqa lvis 'fairscale>=0.4.4' fvcore urllib3 boto3 omegaconf transformers==4.33.2 pycocoevalcap
echo VBENCH_BASE_DONE
