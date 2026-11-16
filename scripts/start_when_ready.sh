#!/usr/bin/env bash
# Launch the 250M run the moment the encoded corpus is complete.
# Deliberately independent of any agent turn: it polls on its own, then execs
# the supervisor with setsid so it survives the parent shell exiting.
set -u
cd "$(dirname "$0")/.."
BIN=data/bin_250m
LOG=reports/startup.log
mkdir -p reports artifacts/run250
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "waiting for $BIN/train.bin + val.bin + corpus_stats.json"
for i in $(seq 1 720); do   # up to 4 hours
  if [ -f "$BIN/train.bin" ] && [ -f "$BIN/val.bin" ] && [ -f "$BIN/corpus_stats.json" ] \
     && ! pgrep -f "encode_stream.py" >/dev/null; then
    say "encoding complete: $(cat "$BIN/corpus_stats.json" | tr -d '\n ')"
    break
  fi
  sleep 20
done

if [ ! -f "$BIN/val.bin" ]; then say "ERROR: val.bin never appeared"; exit 1; fi

# free the 1.9G of text now that it is tokenised
du -sh data/curriculum >> "$LOG" 2>/dev/null
rm -rf data/curriculum/part_*.txt
say "deleted text parts; $(df -h /home/user | tail -1 | awk '{print $4}') free"

if [ -f artifacts/run250/supervisor.pid ] && kill -0 "$(cat artifacts/run250/supervisor.pid)" 2>/dev/null; then
  say "supervisor already running"; exit 0
fi

say "launching supervisor: 61,036 steps = 250,000,000 tokens"
setsid nohup python3 scripts/supervise_run.py \
  --steps 61036 --budget-tokens 250000000 \
  --out artifacts/run250 --bin-dir "$BIN" \
  --batch 16 --ctx 256 --d-model 288 --layers 8 --heads 8 --d-ff 880 \
  --lr 1e-3 --warmup 2000 --threads 2 --eval-every 4000 --ckpt-every 2000 \
  > /tmp/supervisor250.out 2>&1 &
say "supervisor launched pid $!"
