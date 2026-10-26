# Orin power and fan profiles

The Orin NX sits in the living room, and at full power the fan is loud enough to be a real problem. Since 2026-10-25 the owner keeps it on a **quiet default**. A **benchmark profile** is only for runs whose numbers depend on clocks.

## Profiles

| Profile | When | nvpmodel | jetson_clocks | Fan |
|---|---|---|---|---|
| **Quiet (default)** | everything else: offline analysis, gradient or feature extraction, report scripts, idle | `25W` (mode 3) | off | cool mode, as set by the owner |
| **Benchmark** | vla.cpp or any latency/throughput benchmark, chunk-latency measurements, real-arm rollouts and robot days | `MAXN_SUPER` (mode 0) | on | follows jetson_clocks (fan at max) |

Mode IDs come from `/etc/nvpmodel.conf`: 0 MAXN_SUPER, 1 10W, 2 15W, 3 25W, 4 40W. `/etc/nvfancontrol.conf` names two fan profiles, `quiet` and `cool`; its `FAN_DEFAULT_PROFILE` reads `quiet`.

## Commands

Check the state first; this needs no root:

```bash
nvpmodel -q                                  # current power mode
jetson_clocks --show 2>/dev/null || sudo jetson_clocks --show
```

Switch to the benchmark profile before the run:

```bash
sudo jetson_clocks --store ~/jetson_clocks_quiet.conf   # save the quiet clock state once
sudo nvpmodel -m 0                                       # MAXN_SUPER (may ask to reboot when leaving some modes)
sudo jetson_clocks                                       # lock clocks at max, fan to max
```

Switch back to the quiet default straight after the run:

```bash
sudo jetson_clocks --restore ~/jetson_clocks_quiet.conf
sudo nvpmodel -m 3                                       # 25W
sudo systemctl restart nvfancontrol                      # fan back under nvfancontrol (cool / quiet profile)
```

## Rules for agents

- **Exception to AGENTS.md §1 (owner, 2026-10-25):** an agent may use `sudo` for the power and fan commands on this page, and for nothing else: `nvpmodel`, `jetson_clocks`, `systemctl restart nvfancontrol`. It switches to the benchmark profile only when the run needs it, says so in the session (which profile, why, roughly how long), and switches back to the quiet default as soon as the run ends, including when the run fails.
- Record the profile with every timing number: `nvpmodel -q` and the jetson_clocks state go in the protocol, README or summary.json of a benchmark. A latency measured at 25W without jetson_clocks is not comparable with the MAXN_SUPER + jetson_clocks numbers in `projects/vlacpp/jetson-orin-nx.md` or the gate latency files, and it is labelled as such.
- Accuracy-only offline work does not need the benchmark profile: proxies, world-model reports, curation scores, determinism checks. It runs on the quiet default; it is just slower.
- Return the Orin to the quiet default after every benchmark or robot day.
