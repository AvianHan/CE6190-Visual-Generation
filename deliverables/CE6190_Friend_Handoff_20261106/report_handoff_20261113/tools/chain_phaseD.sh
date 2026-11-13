#!/bin/bash
# Per-GPU hand-over: when the grid worker on GPU g reports finished_no_work and its process is gone,
# start the Phase D ablation batch_worker on that GPU (via gpu_env.sh). Ablation results go to results/
# (unique run_ids; reference_run_id points at the same-GPU-model grid baselines). State: state/manifest_ablation.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
g=$1
HB=$W/state/manifest/workers/gpu$g.json
while true; do
  st=$(python3 -c "import json;print(json.load(open('$HB')).get('status',''))" 2>/dev/null)
  pid=$(python3 -c "import json;print(json.load(open('$HB')).get('pid',''))" 2>/dev/null)
  if [ "$st" = "finished_no_work" ] && ! kill -0 "$pid" 2>/dev/null; then break; fi
  if [ -f $W/state/STOP_CHAIN ]; then echo "$(date +%T) stop flag"; exit 0; fi
  sleep 30
done
sleep 5
echo "$(date '+%F %T') grid worker gpu$g finished; starting Phase D batch_worker"
exec $W/tools/gpu_env.sh $W/venv/bin/python $W/tools/batch_worker.py --work $W --gpu $g \
  --manifest $W/phaseD/manifest_ablation.jsonl --results $W/results
