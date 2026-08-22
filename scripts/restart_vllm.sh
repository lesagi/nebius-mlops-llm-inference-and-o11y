#!/usr/bin/env bash
# Restart vLLM and block until it is serving. Args are passed through to
# start_vllm.sh, to sweep one flag at a time in Phases 1 and 6:
#   ./scripts/restart_vllm.sh --max-num-seqs 64
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default the log into the repo, not /tmp: /tmp gets cleaned between sessions and
# the log is the record of which flags a given measurement ran with.
LOG="${VLLM_LOG:-$ROOT/logs/vllm.log}"
mkdir -p "$(dirname "$LOG")"

pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
# On H100, wait for the GPU to actually be released; a new server would
# otherwise fail its memory profiling against the dying process's allocation.
if command -v nvidia-smi > /dev/null 2>&1; then
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [[ "$used" -lt 2000 ]] && break
        sleep 2
    done
fi

cd "$ROOT" || exit 1
set -a
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    . ./.env
fi
set +a
nohup ./scripts/start_vllm.sh "$@" > "$LOG" 2>&1 &

for _ in $(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready"
        grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1
        exit 0
    fi
    if grep -qiE "^(Traceback|.*Error:)" "$LOG" && ! pgrep -f "vllm serve" > /dev/null; then
        echo "vLLM failed to start:"; tail -25 "$LOG"; exit 1
    fi
    sleep 5
done
echo "timed out waiting for vLLM"; tail -25 "$LOG"; exit 1
