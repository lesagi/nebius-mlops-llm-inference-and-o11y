#!/usr/bin/env bash
#
# Start vLLM with a Mac development or H100 production profile.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${VLLM_PROFILE:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        VLLM_PROFILE="mac"
    else
        VLLM_PROFILE="h100"
    fi
fi

case "$VLLM_PROFILE" in
    mac|h100)
        ;;
    *)
        echo "Unsupported VLLM_PROFILE '$VLLM_PROFILE' (expected mac or h100)" >&2
        exit 1
        ;;
esac

# .env contains shared secrets and service settings. An optional profile file
# overrides only machine-specific values such as the model checkpoint.
set -a
if [[ -f "$ROOT/.env" ]]; then
    # shellcheck disable=SC1091
    . "$ROOT/.env"
fi
: "${VLLM_MODEL:?Set VLLM_MODEL in .env - see REPORT.md 'Running this'}"

# No --served-model-name alias: the API id is the checkpoint id, so Langfuse traces and the
# vLLM Prometheus model_name label say which quantization actually served a request.
# The bf16 control (REPORT.md 2.1) means editing VLLM_MODEL in .env; clients follow it.
common_args=(
    --host 0.0.0.0
    --max-num-batched-tokens 4096
    --enable-prefix-caching
)

# Phase 6 changed this block, so it is no longer byte-identical to what
# results/phase1_sweep.json was measured with. Those rows predate
# --api-server-count and describe a saturated single-process front end: treat
# them as historical, not reproducible from here. Every other flag is unchanged,
# so the per-flag reasoning in REPORT.md 1 still applies.
if [[ "$VLLM_PROFILE" == "mac" ]]; then
    profile_args=(
        --max-model-len 4096
        --max-num-seqs 8
    )
else
    profile_args=(
        --max-model-len 8192
        --gpu-memory-utilization 0.90
        --max-num-seqs 256
        --async-scheduling
        --disable-log-requests
        # One API server process could not keep up: 87% of end-to-end latency was
        # spent between the front end and the engine scheduler, and /metrics itself
        # took 17s+ (often timing out), blinding Prometheus under load. Four
        # processes took mean e2e from 17.2s to 0.90s at higher throughput, with
        # that unaccounted share collapsing to 0.7%. Never swept: 4 was the first
        # value tried and it already cleared the SLO.
        # 8, not 4, because every Phase 6 run was served at 8 and this script has to
        # reproduce the config those numbers came from. The 8 arrived by accident -
        # see the passthrough note below - and the front end never queued at either
        # value (unaccounted 0.01s, 100% scrape success), so this is about matching
        # the measurement, not about 8 being better.
        --api-server-count 8
    )
fi

echo "Starting vLLM with profile=$VLLM_PROFILE model=$VLLM_MODEL"
# "$@" comes last so a sweep wins over the block above - `restart_vllm.sh
# --max-num-seqs 64` needs no edit here. The catch, which cost real confusion in
# Phase 6: for a flag already in profile_args this passes it *twice* and argparse
# silently takes the last. That is how the served api-server-count drifted from 4
# to 8 without the script changing. scripts/phases/phase6.py greps the launch log
# for the effective value rather than trusting this file, which is how it surfaced.
exec uv run vllm serve "$VLLM_MODEL" \
    "${common_args[@]}" \
    "${profile_args[@]}" \
    "$@"
