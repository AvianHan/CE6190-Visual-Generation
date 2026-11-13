#!/bin/bash
# Build detectron2 (pinned commit) into the separate VBench env only; generation env untouched.
set -ex
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16 FORCE_CUDA=1
cd /home/ecs-user/ce6190-runtime/a100x8-20261110/vendor/detectron2
git checkout -q --detach 247691a1c292654867deb35cabb27f18d75a1363
git log -1 --format='%H %cd'
nice -n 19 /home/ecs-user/ce6190-runtime/a100x8-20261110/venv_vbench/bin/pip install --no-build-isolation -e .
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv_vbench/bin/python -c "import detectron2, detectron2._C; print('detectron2', detectron2.__version__, 'ext ok')"
echo D2_DONE
