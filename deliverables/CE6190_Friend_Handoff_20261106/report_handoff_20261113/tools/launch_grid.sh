#!/bin/bash
# Launch one detached worker per GPU for the handoff grid manifest.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
mkdir -p $W/logs/workers
for g in 0 1 2 3 4 5 6 7; do
  setsid nohup $W/venv/bin/python $W/tools/worker.py --work $W --gpu $g > $W/logs/workers/grid_gpu$g.log 2>&1 < /dev/null &
  echo "gpu $g worker pid $!"
  sleep 2
done
