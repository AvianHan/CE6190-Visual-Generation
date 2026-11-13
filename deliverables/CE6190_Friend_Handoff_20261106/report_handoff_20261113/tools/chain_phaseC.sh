#!/bin/bash
# Per-GPU hand-over Phase D -> Phase C (frozen 1.6x test). Starts only when (a) this GPU's Phase D batch_worker
# reports finished_no_work and has exited, and (b) it is at/after NOT_BEFORE (window for the user to change the
# freeze target / seed count). Stop flag: state/STOP_CHAIN_C.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
g=$1
NOT_BEFORE=$(date -d '2026-11-12 07:00:00' +%s)
HB=$W/state/manifest_ablation/workers/gpu$g.json
while true; do
  if [ -f $W/state/STOP_CHAIN_C ]; then echo "$(date +%T) stop flag"; exit 0; fi
  st=$(python3 -c "import json;print(json.load(open('$HB')).get('status',''))" 2>/dev/null)
  pid=$(python3 -c "import json;print(json.load(open('$HB')).get('pid',''))" 2>/dev/null)
  if [ "$st" = "finished_no_work" ] && ! kill -0 "$pid" 2>/dev/null && [ $(date +%s) -ge $NOT_BEFORE ]; then break; fi
  sleep 30
done
sleep 5
echo "$(date '+%F %T') Phase D on gpu$g finished; starting Phase C batch_worker"
exec $W/tools/gpu_env.sh $W/venv/bin/python $W/tools/batch_worker.py --work $W --gpu $g \
  --manifest $W/phaseC/manifest_test.jsonl --results $W/results
