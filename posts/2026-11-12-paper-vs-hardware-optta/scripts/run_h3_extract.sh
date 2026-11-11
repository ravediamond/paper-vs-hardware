#!/usr/bin/env bash
# WS-196 H3 extraction in the lerobot-jetson-wm image (W1=1 layout of shared/gate/jetson_run.sh) with --memory 8g,
# GPU lock held only while a docker run is alive, hard timeouts, at most 2 retries (3 attempts), HF offline.
#   projects/optta/run_h3_extract.sh [cpu] [extra extract args...]   ("cpu" hides the GPU: the lock is still taken)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$HOME/scratch/WS-196/h3_lat}
LOG=${LOG:-$REPO/journal/workstack/runs/WS-196/h3_extract.log}
TOTAL=${TOTAL:-5400}      # seconds per attempt
mkdir -p "$(dirname "$LOG")" "$OUT"
GPU=(--runtime nvidia); ENVX=()
if [ "${1:-}" = cpu ]; then shift; ENVX=(-e CUDA_VISIBLE_DEVICES=); fi
for attempt in 1 2 3; do
  echo "$(date +%F\ %T) attempt $attempt/3" | tee -a "$LOG"
  flock -w 7200 ~/scratch/gpu.lock timeout -k 30 "$TOTAL" docker run --rm -i "${GPU[@]}" --memory 8g --memory-swap 8g --network host --ipc host \
    "${ENVX[@]}" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/wm_data:$HOME/wm_data:ro" \
    -v "$HOME/gate/pydeps-w1:/opt/pydeps-w1:ro" -v "$HOME/gate/pydeps:/opt/gate-pydeps:ro" \
    -v "$REPO:/work" -v "$OUT:/out" -w /work -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/opt/pydeps-w1:/work/shared/gate:/opt/gate-pydeps lerobot-jetson-wm:rc \
    bash -c 'python projects/optta/extract_w1_h3.py --out /out "$@"; rc=$?; chown -R '"$(id -u):$(id -g)"' /out; exit $rc' _ "$@" >>"$LOG" 2>&1
  rc=$?
  echo "$(date +%F\ %T) attempt $attempt exit $rc" | tee -a "$LOG"
  grep -q "EXTRACTION COMPLETE" "$LOG" && [ $rc -eq 0 ] && exit 0
done
echo "extraction failed after 3 attempts (1 + 2 retries): stop and record" | tee -a "$LOG"; exit 1
