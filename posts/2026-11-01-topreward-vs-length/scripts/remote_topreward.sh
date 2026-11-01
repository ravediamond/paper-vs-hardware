#!/bin/bash
# On the vast.ai instance (uploaded as /root/wm/run_remote.sh; started by wm_train/vast/bootstrap.sh under its hard
# deadline). TOPREWARD_PROTOCOL.md: pinned model and data (public, anonymous), own 69 first, then ArmnetBench.
set -uo pipefail
W=/root/wm; O=$W/out; mkdir -p $O; cd $W
MODEL_REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
AB=armnet/armnetbench_v01_lerobot_so101; AB_REV=1db69c82b589c7172ee31a6b8f7ef58d1fc2c663
OWN=ravediamond/so101-bowls-policy-rollouts; OWN_REV=bad1c8ea363789f5f55f9098e775f7361a0f019f
t0=$(date +%s); stamp() { echo "$1=$(( $(date +%s) - t0 ))" >> $O/timings.env; echo "$(date -Is) $1"; }
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $O/gpu.log; sleep 60; done ) &
apt-get update -q && apt-get install -y -q --no-install-recommends ca-certificates curl > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.12 $W/venv && . $W/venv/bin/activate
uv pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "transformers==5.5.4" accelerate av pandas pyarrow "huggingface_hub[hf_transfer]" torchvision --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" | tee $O/env.txt
stamp env_ready
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-8B-Instruct", revision="$MODEL_REV", local_dir="$W/model", token=False)
snapshot_download("$OWN", repo_type="dataset", revision="$OWN_REV", local_dir="$W/own", token=False,
                  allow_patterns=["meta/*", "meta/**", "videos/observation.images.front_top_camera/**"])
snapshot_download("$AB", repo_type="dataset", revision="$AB_REV", local_dir="$W/ab", token=False,
                  allow_patterns=["meta/*", "meta/**", "videos/observation.images.top/**"])
PYEOF
stamp data_ready; df -h /root | tail -1
python $W/topreward_score.py --root $W/own --view front_top_camera --labels $W/own/meta/rollout_labels.csv --model $W/model --out $O/own.jsonl 2>&1 | grep -v -i warn | tail -5; stamp own
python $W/topreward_score.py --root $W/ab --view top --armnet --model $W/model --out $O/ab.jsonl 2>&1 | grep -v -i -E "warn|Loading" ; stamp ab
touch $O/DONE
