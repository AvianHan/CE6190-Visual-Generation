#!/bin/bash
set -e
while ! grep -q TORCH_DONE /home/ecs-user/ce6190-runtime/a100x8-20261110/logs/env_setup.log; do
  if grep -qE "^ERROR|Error:" /home/ecs-user/ce6190-runtime/a100x8-20261110/logs/env_setup.log; then echo "torch install failed"; exit 1; fi
  sleep 10
done
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip install -r /home/ecs-user/ce6190-runtime/a100x8-20261110/code/requirements.txt
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip install einops
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip install --no-deps -e /home/ecs-user/ce6190-runtime/a100x8-20261110/code
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip check
/home/ecs-user/ce6190-runtime/a100x8-20261110/venv/bin/pip freeze > /home/ecs-user/ce6190-runtime/a100x8-20261110/logs/pip-freeze.txt
echo REST_DONE
