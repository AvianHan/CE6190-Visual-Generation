#!/bin/bash
# Run a command with the generation venv's own NVIDIA libraries only (no host CUDA/cuDNN on the loader path).
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
export LD_LIBRARY_PATH=$(ls -d $W/venv/lib/python3.11/site-packages/nvidia/*/lib | sort | paste -sd:)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$@"
