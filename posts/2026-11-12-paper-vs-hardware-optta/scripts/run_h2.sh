#!/usr/bin/env bash
# H1/H2 driver: one torch job at a time under the memory cap; skips a finished (method, seed). Detach with nohup.
cd "$(dirname "$0")/../.." || exit 1
export WS=WS-196
PY=$HOME/venvs/optta/bin/python
R=projects/optta/results
run() { # method seed n out-extra
  out=$R/h2_$1_s$2.json
  if [ -f "$out" ] && $PY -c "import json,sys;d=json.load(open('$out'));sys.exit(0 if len(d['episodes'])>=$3 else 1)"; then return; fi
  timeout 21600 scripts/memcap.sh -m 11G $PY projects/optta/optta_pusht.py --method $1 --seed $2 --n $3 --out $out \
    > journal/workstack/runs/WS-196/h2_$1_s$2.log 2>&1
}
for s in 0 1 2; do run frozen $s 50; run adajepa $s 50; run optta $s 50; done
run pess 0 50
run optta30 0 5
