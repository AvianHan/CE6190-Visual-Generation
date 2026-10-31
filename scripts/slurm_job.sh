#!/usr/bin/env bash
set -euo pipefail
# Activate the prepared environment before submission. Slurm exports it by default.
# Site-specific module/account/partition settings belong in submit_slurm.py arguments.
: "${SLURM_ARRAY_TASK_ID:?This script must run inside a Slurm job array}"
python -m cachebench.generate --manifest "$1" --index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$2" --wan-repo "$3" --vendor-dir "$(dirname "$3")" --checkpoint-dir "$4"
