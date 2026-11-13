#!/bin/bash
# Waits for Phase C to finish (3750 done records, all 8 batch_workers exited), runs acceptance over every run, and
# starts the VBench pipeline only if acceptance reports 0 bad. Stop flag: state/STOP_AFTER_C.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110; cd $W
while true; do
  [ -f state/STOP_AFTER_C ] && { echo "$(date +%T) stop flag"; exit 0; }
  n=$(ls state/manifest_test/done | wc -l)
  w=$(ps -eo cmd | grep -c '^/[^ ]*python [^ ]*tools/batch_worker.py')
  [ "$n" -ge 3750 ] && [ "$w" -eq 0 ] && break
  sleep 60
done
echo "$(date '+%F %T') Phase C finished ($n done); acceptance"
nice -n 5 venv/bin/python tools/accept_grid.py phaseC/manifest_test.jsonl acceptance/phaseC_check.jsonl > logs/accept_phaseC.log 2>&1
tail -1 logs/accept_phaseC.log
bad=$(python3 -c "import json;print(json.loads(open('logs/accept_phaseC.log').read().splitlines()[-1])['bad'])" 2>/dev/null)
nc=$(python3 -c "
import json,os;d='state/manifest_test/done';print(sum(json.load(open(f'{d}/{f}'))['status']!='complete' for f in os.listdir(d)))")
if [ "$bad" != "0" ] || [ "$nc" != "0" ]; then echo "$(date +%T) acceptance bad=$bad non-complete=$nc -> NOT starting VBench"; exit 1; fi
echo "$(date '+%F %T') acceptance ok; starting VBench pipeline"
exec venv/bin/python tools/vbench_phaseC.py > logs/vbench_phaseC.log 2>&1
