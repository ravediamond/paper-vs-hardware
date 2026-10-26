#!/usr/bin/env bash
# predownload done -> stop vla-server -> cool down (Tj < 60 C) -> full sweep. Logs to ~/vla-bench-results/chain.log.
L=~/vla-bench-results/chain.log
log() { echo "$(date "+%F %T") $*" >> $L; }
log "waiting for predownload"
until grep -q "predownload finished" ~/vla-bench-results/orin_nx.log; do sleep 30; done
log "predownload finished; stopping vla-server"
pkill -x vla-server; sleep 5
pgrep -x vla-server >/dev/null && log "WARNING: vla-server still running"
pgrep -x tegrastats >/dev/null && log "tegrastats logging: yes" || log "WARNING: tegrastats not running"
for i in $(seq 1 90); do
  T=$(tail -1 ~/vla-prof/run.tegrastats | grep -oE "tj@[0-9.]+" | cut -d@ -f2)
  awk -v t="$T" "BEGIN{exit !(t<60)}" && break
  sleep 20
done
log "starting sweep at Tj=${T}C, $(df -h / | tail -1 | awk "{print \$4}") free"
python3 ~/bench_sweep.py >> ~/vla-bench-results/sweep.out 2>&1
log "sweep exited with $?"
