#!/usr/bin/env bash
set -euo pipefail

TOTAL_RUNS=2                         # << set to 1000 for one thousand runs
RUN_TIME_PER_GAME_SEC="${RUN_TIME_PER_GAME_SEC:-90}"  # << set default seconds per run it needs to be 500

mkdir -p logs bin

for ((i=1; i<=TOTAL_RUNS; i++)); do
  echo "========== RUN $i / $TOTAL_RUNS =========="
  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  LOGFILE="logs/run_${ts}.log"

  echo "[INFO] compiling sources..."
  find src -name '*.java' > sources.list
  javac -cp "lib/*:bin" -d bin @sources.list

  echo "[INFO] starting game (will auto-stop after ${RUN_TIME_PER_GAME_SEC}s)..."
  java -cp "lib/*:bin" rts.MicroRTS >"$LOGFILE" 2>&1 &
  game_pid=$!

  sleep "$RUN_TIME_PER_GAME_SEC"

  if kill -0 "$game_pid" 2>/dev/null; then
    kill "$game_pid" 2>/dev/null || true
    sleep 2
    kill -0 "$game_pid" 2>/dev/null && kill -9 "$game_pid" 2>/dev/null || true
  fi
  wait "$game_pid" 2>/dev/null || true
  echo "[INFO] run $i complete."
  echo
done

echo "ALL $TOTAL_RUNS RUNS DONE 🎉"