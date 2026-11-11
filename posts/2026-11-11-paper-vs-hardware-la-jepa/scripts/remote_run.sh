#!/usr/bin/env bash
# WS-167, runs ON the vast instance (started by scripts/vast_lajepa.sh). Layout in /root/l: vjepa2/ (facebookresearch/vjepa2 @ 204698b), proj/{lajepa,gawm}/ (this repo's folders).
#   remote_run.sh setup : install, dataset (public, pinned revision) + decode (projects/gawm/prepare.py), V-JEPA 2.1 ViT-L weights, FLAN-T5 text embeddings,
#                         encoder sanity (stored vs re-encoded block), feature store (build, stats, check), speed probe (3 parallel runs of 60 steps)
#   remote_run.sh runs  : 6 runs = {full (fut weight 1), nofuture (fut weight 0)} x seeds 0,1,2; the 3 seeds of one config in parallel
# Results: /root/l/out/<cfg>_s<seed>/ (heldout_mae.json, train_log.jsonl, run.json; model.pt stays unless pulled).
set -uo pipefail
cd /root/l
STEPS=${STEPS:-20000}
REV=4e5c6fda8dc5ce2252214bc086181d3df8b594b0
P=/root/l/proj/lajepa
export VJEPA2_REPO=/root/l/vjepa2 VJEPA2_CKPT=/root/l/vjepa2_1_vitl_dist_vitG_384.pt
mkdir -p out probe
if [ "${1:-}" = setup ]; then
  pip install -q timm einops "transformers==4.57.0" sentencepiece av opencv-python-headless pandas pyarrow huggingface_hub safetensors 2>&1 | tail -2
  [ -d vjepa2/.git ] || git clone -q https://github.com/facebookresearch/vjepa2 vjepa2
  git -C vjepa2 checkout -q 204698b45b3712590f06245fbfba32d3be539812 || { echo "vjepa2 checkout failed"; exit 1; }
  [ -f "$VJEPA2_CKPT" ] || wget -q -O "$VJEPA2_CKPT" https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt || { echo "ckpt download failed"; exit 1; }
  python3 - <<P
from huggingface_hub import snapshot_download
print(snapshot_download("ravediamond/so101_bowls_all_trimmed", repo_type="dataset", revision="$REV", local_dir="/root/l/raw"))
P
  [ -f data/.prepared ] || { python3 proj/gawm/prepare.py --root /root/l/raw --out /root/l/data && touch data/.prepared; } || { echo "prepare failed"; exit 1; }
  df -h /root | tail -1; nproc; free -g | sed -n 2p; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python3 $P/features.py text --data data --out store || { echo "text failed"; exit 1; }
  # encoder sanity on 2 episodes before the bulk encode: shapes assert inside encode(); finite; token statistics
  if [ ! -f store_chk/.ok ]; then
    python3 proj/gawm/prepare.py --root /root/l/raw --out /root/l/data_chk --episodes 0,172 > /dev/null || { echo "prepare chk failed"; exit 1; }
    python3 $P/features.py build --data data_chk --out store_chk --encoder vjepa21 --stride 8 || { echo "encoder sanity build failed"; exit 1; }
    python3 $P/features.py check --data data_chk --out store_chk || { echo "encoder sanity check failed"; exit 1; }
    touch store_chk/.ok
  fi
  python3 $P/features.py build --data data --out store --encoder vjepa21 --stride 2 --resume || { echo "build failed"; exit 1; }
  python3 $P/features.py stats --data data --out store && python3 $P/features.py check --data data --out store
  df -h /root | tail -1
  for s in 0 1 2; do
    python3 $P/train.py --data data --store store --seed $s --steps 60 --workers 4 --log-every 20 --eval-limit 64 --out probe/s$s > probe/s$s.log 2>&1 &
  done
  T0=$(date +%s); wait; T1=$(date +%s)
  echo "PROBE wall $((T1 - T0)) s for 3 x 60 steps incl. start-up and eval"
  tail -n 1 probe/s0/train_log.jsonl
  echo "PROBE_DONE"
  exit 0
fi
run() { # config fut_weight seed
  python3 $P/train.py --data data --store store --fut-weight "$2" --seed "$3" --steps "$STEPS" --workers 4 --out "out/$1_s$3" > "out/$1_s$3.log" 2>&1
}
for cfg in "full 1.0" "nofuture 0.0"; do
  set -- $cfg
  [ -f out/$1_s0/heldout_mae.json ] && [ -f out/$1_s1/heldout_mae.json ] && [ -f out/$1_s2/heldout_mae.json ] && { echo "skip $1 (done)"; continue; }  # restart after a container loss
  for s in 0 1 2; do run "$1" "$2" $s & done
  wait
  echo "done $1 $(date +%s)"
done
echo ALL_DONE
