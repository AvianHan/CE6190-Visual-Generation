#!/bin/bash
# Five-method full-protocol check on one pair (dev_04_04 seed0), one task per GPU, in parallel.
W=/home/ecs-user/ce6190-runtime/a100x8-20261110
mkdir -p $W/logs/smoke2
NVLIB=$(ls -d $W/venv/lib/python3.11/site-packages/nvidia/*/lib | sort | paste -sd:)
gpu=0
for idx in 151 153 156 160 163; do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$gpu '$1==g{print $2}')
  LD_LIBRARY_PATH=$NVLIB CUDA_VISIBLE_DEVICES=$uuid OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    nohup $W/venv/bin/python -m cachebench.generate --manifest $W/manifest.jsonl --index $idx \
    --output-dir $W/results --vendor-dir $W/vendor --checkpoint-dir $W/models/Wan2.1-T2V-1.3B --device 0 \
    > $W/logs/smoke2/idx$idx-gpu$gpu.log 2>&1 &
  echo "idx $idx -> gpu $gpu ($uuid) pid $!"
  gpu=$((gpu+1))
done
