# Phase runners

One script per phase. Each verifies its own deliverable and then does whatever
the phase needs a human present for (taking a screenshot, reading a dashboard).

Run from the repo root:

```bash
uv run python scripts/phases/phase2.py
```

| Script | Phase | What it does | Deliverable it sets up |
|---|---|---|---|
| `phase2.py` | 2 | Checks Prometheus/Grafana/vLLM are up, validates every PromQL expression in `serving.json` against the live TSDB, then drives load so the panels move. | `screenshots/grafana_serving.png` |
| `phase3.py` | 3 | Asserts the router truth table, the verdict parser and the SQL extractor offline, then starts the agent on :8001 and runs eval-set questions through it, printing each run's node path. | none - prints the `verify → revise` evidence |
| `phase4.py` | 4 | Fires tagged questions through the agent, then reads the traces back out of Langfuse to confirm they are tagged, complete and carry the node waterfall. | `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png` |
| `phase5.py` | 5 | Asserts the eval scoring offline, then drives `evals/run_eval.py` over the 30-question set and prints the overall, set-wise and per-iteration pass rates. | `results/eval_baseline.json`, `screenshots/grafana_eval_run.png` |
| `phase6.py` | 6 | Drives `load_test/driver.py` at the SLO arrival rate, then puts the client-side percentiles next to vLLM's own view of the same traffic and records both against the configuration they ran under. | `results/phase6_slo.json`, `screenshots/grafana_before.png`, `screenshots/grafana_after.png` |

Later phases get their scripts as they are built.

## phase2.py

```bash
uv run python scripts/phases/phase2.py                 # verify, then 5 min of load
uv run python scripts/phases/phase2.py --verify-only   # checks only, no load
uv run python scripts/phases/phase2.py --rps 10 --duration 300
```

Defaults to `--rps 0.5` on a Mac (CPU vLLM) and `--rps 10` on the H100, which is
the SLO arrival rate. Take the screenshot while the load is running — the script
prints the dashboard URL and tells you when to capture.

Resolves `VLLM_MODEL` from `.env` before invoking
`load_test/vllm_probe.py`, which only reads `os.environ`. Without that the probe
falls back to the hard-coded H100 checkpoint id and vLLM answers an unserved
model with a bare `404` that looks like a wrong URL.

Report anything from a real run against `Qwen3-30B-A3B` on the H100 — CPU-vLLM
numbers are unrepresentative and only prove the panels are wired correctly.

## phase3.py

```bash
uv run python scripts/phases/phase3.py                 # asserts + 5 live questions
uv run python scripts/phases/phase3.py --verify-only   # asserts only, no backend
uv run python scripts/phases/phase3.py -n 10 --timeout 600
```

`--verify-only` needs nothing running: it checks `route_after_verify`,
`_parse_verdict` and `_extract_sql` with plain `assert`s. The live half needs
vLLM on :8000; it starts `uvicorn agent.server:app` itself if :8001 is quiet,
logs it to `logs/agent.log`, and stops it again afterwards.

Per-question output is the node path (`generate_sql → verify → revise → …`),
iteration count, wall time, and the final SQL, which is where the Phase 3
"at least one question triggers a revise" evidence comes from.

Against the CPU stand-in this only proves the graph is wired. `Qwen3-0.6B` is a
hybrid model that reasons out loud, so it spends its 512-token budget on
`<think>` and often gets truncated before emitting SQL, and its verifier is
too lenient to catch a wrong column. Prompt quality has to be judged against
`Qwen3-30B-A3B-Instruct-2507` on the H100, which does not think.

## phase4.py

```bash
uv run python scripts/phases/phase4.py                 # 10 tagged runs, then verify
uv run python scripts/phases/phase4.py --verify-only   # stack + stored traces, fires nothing
uv run python scripts/phases/phase4.py -n 3
```

Reuses `phase3.py`'s agent process management, so it will start `uvicorn` on
:8001 if nothing is listening. Backend-agnostic — trace shape doesn't depend on
model quality — but 10 questions against the CPU stand-in takes 10-15 minutes.

The verification half calls Langfuse's own API rather than trusting that the
callback fired: it flushes, lists traces filtered on the `phase:4` tag, and
fails if any recent trace has no root span. That empty-name case is the exact
symptom of an unflushed shutdown, which truncated the last trace of every
Phase 3 run before `agent/server.py` grew its lifespan flush.

Tags reach Langfuse as flat `key:value` strings because a Langfuse tag is a
string, not a pair. `/answer` takes them as a dict, `_trace_config` in
`agent/server.py` flattens them into `langfuse_tags` and keeps the dict in trace
metadata. Only the root run's metadata is read, and only with the exact types
the SDK expects — a dict where it wants a list is dropped silently.

## phase5.py

```bash
uv run python scripts/phases/phase5.py                 # asserts + all 30 questions
uv run python scripts/phases/phase5.py --verify-only   # asserts only, no backend
uv run python scripts/phases/phase5.py --limit 3       # smoke test
uv run python scripts/phases/phase5.py --report-only --out results/eval_baseline.json
```

`--verify-only` covers the two pieces of Phase 5 logic that need no model: the
per-iteration carry-forward in `summarize()` (a question that terminated at
iteration 1 must keep that answer at iteration 2, or the series reads as a
regression it never had) and the multiset/set-wise split. The live half starts
the agent itself if :8001 is quiet and shells out to `evals/run_eval.py`, the way
`phase2.py` shells out to `vllm_probe.py`.

30 questions is ~5 minutes on the H100 and ~40 against a CPU stand-in, hence
`--limit`. Take `screenshots/grafana_eval_run.png` while it runs — the script
prints the URL first.

Two pass rates are reported because the provided `canonicalize` compares row
*multisets*, so a correct query that omits `DISTINCT` fails it while BIRD's own
eval, which compares sets, would pass it. The gap between the two is the count
of DISTINCT-class disagreements. Set-wise is looser rather than more correct —
gold question 30 legitimately returns duplicate rows that it would let a
prediction collapse — so neither number is dropped.

The verifier 2×2 is what makes a flat per-iteration series diagnosable: a
`loop_delta` of zero with accepted-wrong answers means the verifier never fired
the loop, which is a different bug from a reviser that fires and cannot fix
anything.

`SCHEMA_SAMPLE_VALUES=1` renders low-cardinality text columns with their
distinct values as a comment. Off by default because Phase 5 measured it as a
net loss (30.0% against the baseline's 36.7%):

```bash
uv run python scripts/phases/phase5.py                            # baseline rendering
SCHEMA_SAMPLE_VALUES=1 uv run python scripts/phases/phase5.py \
    --out results/eval_schema_values.json --label schema-values   # with values
```

The flag is read once per process, so it only applies to an agent the script
starts. `/health` reports which rendering the live agent is using and the script
warns when that disagrees with the flag — reusing an already-running agent is
the one way to make the two runs silently incomparable.

## phase6.py

```bash
uv run python scripts/phases/phase6.py                        # 10 rps, 300s, then diagnose
uv run python scripts/phases/phase6.py --rps 20 --label push  # find the break point
uv run python scripts/phases/phase6.py --workers 4 --label workers-4
uv run python scripts/phases/phase6.py --verify-only          # stack + config census only
uv run python scripts/phases/phase6.py --report-only          # the table of recorded runs
```

The SLO is p95 end-to-end **agent** latency under 5s at 10+ full runs/s over a
5-minute window. Nothing before this phase measured that: `vllm_probe.py` replays
calls against vLLM rather than running the agent, and the Phase 5 eval runs
sequentially at concurrency 1.

The agent exports no metrics of its own and does not need to. `driver.py` gives
the client-side percentiles, Prometheus gives vLLM's own per-call latency, and
the **gap between them is the agent's overhead** — FastAPI, LangGraph, sqlite,
the Langfuse callback, and any queueing inside the uvicorn process. The
`agent overhead` line is that subtraction, at p50 because summing p95s is not a
latency. Calls per run is measured rather than assumed: it is 2 when verify
accepts and 5 after two revises, so it depends on the question pool.

Every PromQL expression is lifted verbatim from `serving.json` with
`$__rate_interval` resolved to `5m`, so the printed numbers and the screenshot
cannot disagree.

PASS needs four things, not one: p95 under 5s, the driver actually achieving the
arrival rate it asked for, zero failed requests, and Prometheus capturing ≥95% of
scrapes — a window the dashboard half missed is not evidence of anything. A
missed SLO still exits 0, because that is a finding to diagnose rather than a
broken script.

Each run is appended to `results/phase6_slo.json` together with a config census —
uvicorn workers, `MAX_ITERATIONS`, schema-value rendering, `--api-server-count`,
`--max-num-seqs`, CPU count — because an iteration log whose rows cannot be
attributed to a configuration is worthless. `--report-only` prints them as a
table. The two vLLM flags come from grepping `logs/vllm.log`, since neither is in
`/metrics`; they read `unknown` rather than guessing.

`--workers` only applies to an agent this script starts. If one is already
listening on :8001 the script says so and refuses to pretend, the same way
`phase5.py` guards the schema-values flag.
