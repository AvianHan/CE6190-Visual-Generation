#!/bin/bash
# Low-priority incremental paired fidelity for the Phase C frozen test (references in the same manifest), every 10 min.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
while [ ! -f $W/state/STOP_EVAL ]; do
  echo "[$(date +%F' '%T)] pass"
  nice -n 19 $W/tools/gpu_env.sh $W/venv/bin/python $W/tools/eval_incremental.py --work $W --procs 12 --device cpu \
    --manifest $W/phaseC/manifest_test.jsonl --results $W/results
  sleep 600
done
