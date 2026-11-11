#!/usr/bin/env bash
# WS-168, runs ON the vast instance (started by scripts/vast_gawm.sh). Layout in /root/g: ga-wm/, ref_libero/config.yaml, dino/ (ViT-S/16 snapshot), proj/ (this folder).
#   remote_run.sh setup : install, pull the dataset (public, pinned revision) and decode it, speed probe (3 parallel runs of 60 steps)
#   remote_run.sh runs  : the 12 runs: for gawm/nowm x full (train172) / quarter (quarter_seed0), the three seeds in parallel.
# Results: /root/g/out/<model>_<split>_s<seed>/ (heldout_l1.json, train_log.jsonl, run.json).
set -uo pipefail
cd /root/g
STEPS=${STEPS:-20000}
export GAWM_REPO=/root/g/ga-wm GAWM_REF_CFG=/root/g/ref_libero/config.yaml GAWM_ENC=/root/g/dino
REV=4e5c6fda8dc5ce2252214bc086181d3df8b594b0
mkdir -p out probe
if [ "${1:-}" = setup ]; then
  pip install -q "transformers==4.57.0" omegaconf accelerate scipy pillow "pydantic==2.10.6" numpydantic tyro rich websocket-client websockets \
    opencv-python-headless requests msgpack modelscope av pandas pyarrow huggingface_hub filelock safetensors 2>&1 | tail -2
  python3 - <<E
from huggingface_hub import snapshot_download
print(snapshot_download("ravediamond/so101_bowls_all_trimmed", repo_type="dataset", revision="$REV", local_dir="/root/g/raw"))
E
  python3 proj/prepare.py --root /root/g/raw --out /root/g/data || { echo "prepare failed"; exit 1; }
  df -h /root | tail -1; nproc; free -g | sed -n 2p
  for s in 0 1 2; do
    python3 proj/gawm_so101.py train --data /root/g/data --model gawm --train-split train172 --seed $s --steps 60 --workers 4 --log-every 20 --eval-limit 64 --out probe/s$s > probe/s$s.log 2>&1 &
  done
  T0=$(date +%s); wait; T1=$(date +%s)
  echo "PROBE wall $((T1 - T0)) s for 3 x 60 steps incl. start-up and eval"
  tail -n 1 probe/s0/train_log.jsonl
  echo "PROBE_DONE"
  exit 0
fi
run() { # model split seed
  python3 proj/gawm_so101.py train --data /root/g/data --model "$1" --train-split "$2" --seed "$3" --steps "$STEPS" --workers 4 --out "out/$1_$2_s$3" > "out/$1_$2_s$3.log" 2>&1
}
# TODO_GROUPS ("model:split ..." in run order) lets the launcher resume: groups whose results are already on the Orin are left out. Default: all four.
for g in ${TODO_GROUPS:-gawm:train172 nowm:train172 gawm:quarter_seed0 nowm:quarter_seed0}; do
  m=${g%%:*}; split=${g##*:}
  [ -f out/${m}_${split}_s0/heldout_l1.json ] && [ -f out/${m}_${split}_s1/heldout_l1.json ] && [ -f out/${m}_${split}_s2/heldout_l1.json ] && { echo "skip $m $split (done)"; continue; }  # restart after a container loss
  for s in 0 1 2; do run $m $split $s & done
  wait
  echo "done $m $split $(date +%s)"
done
echo ALL_DONE
