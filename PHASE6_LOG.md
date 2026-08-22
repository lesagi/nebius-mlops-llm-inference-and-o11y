# Phase 6 — SLO iteration log

> **SLO:** p95 end-to-end *agent* latency under 5s, at 10+ full agent runs/s, over a 5-minute window.

One change per iteration. Each entry is written in this order: what the dashboard showed, what
that made me think, the single thing I changed, what actually happened. The hypothesis is
recorded **before** the run, so a wrong one stays on the page.

**Pass needs four things**, not just the p95 (`verdict()` in `scripts/phases/phase6.py`):
p95 < 5s, zero failed requests, achieved rate ≥ 95% of offered (a shortfall *is* a backlog), and
≥ 95% Prometheus scrape success (a window with gaps is not evidence).

Reproduce any row:

```bash
# MAX_ITERATIONS=2 from iteration 2 on; the committed default is 3 (the brief's range).
MAX_ITERATIONS=2 uv run python scripts/phases/phase6.py --rps 10 --duration 300 --label <label>
uv run python scripts/phases/phase6.py --report-only          # the table of every recorded run
uv run python scripts/phases/phase6.py --slowest 5 --label <label>   # per-node, out of Langfuse
```

Raw data: every run appends a summary row to `results/phase6_slo.json` (tracked - it carries every
figure quoted below). Per-request latencies land in `results/phase6_load_<label>.json`, which are
deliberately **not** committed: 4.5 MB of raw dumps that regenerate by re-running a row.

## The layers, and which number sees which

| Number | Where it comes from | What it can prove |
|---|---|---|
| client p50/p95 | `load_test/driver.py` | the SLO itself, and nothing about why |
| `vllm_call_p50/p95`, queue/prefill/decode | Prometheus, vLLM `/metrics` | whether the serving layer is the problem |
| `agent_overhead_p50` = client p50 − (calls/run × vLLM p50) | the two above, subtracted | time inside the agent process that vLLM never saw |
| `cpu.*_cores` | `/proc/[pid]/stat` deltas over the run | who ate the 16 cores |
| per-node seconds | Langfuse spans | *which* part of a run, when Grafana says the GPU is innocent |

`agent_overhead_p50` is the load-bearing one. vLLM's own p95 stayed under 5s in every run below,
including the ones that missed the SLO by 90 seconds — so from iteration 1 on, this is not a
serving-config problem.

---

## Runs from 21 Aug (prior context, all contaminated — see iteration 1)

| label | rps | p50 | p95 | calls/run | overhead p50 | box cores busy | verdict |
|---|---|---|---|---|---|---|---|
| workers-4 | 8.3 | 5.45 | 97.88 | 3.0 | 2.89 | n/a | FAIL |
| workers-2 | 8.3 | 1.92 | 111.80 | 2.0 | 0.55 | 11.7 | FAIL |
| rps-5 | 4.2 | 1.78 | 21.89 | 3.1 | 0.31 | 13.5 | FAIL |
| rps-3 | 3.0 | 1.11 | 7.66 | 2.9 | −0.10 | 12.5 | FAIL |
| iters-2 | 3.0 | 1.21 | 6.79 | 2.2 | 0.36 | 11.5 | FAIL |
| query-budget | 8.3 | 1.86 | 94.98 | 1.5 | 0.92 | 9.5 | FAIL |
| rps-3-budget | 3.0 | 0.82 | 3.88 | 3.1 | −0.33 | 10.1 | PASS |

What that table said before iteration 1: the system holds the SLO at 3 rps and collapses somewhere
between 3 and 5. **It does not collapse because of vLLM** — the worst row above still had
`vllm_call_p95` 4.77s, `queue_mean` 27µs, KV at 0.7%, zero preemptions. The GPU was coasting while
the client saw 98s.

The suspicious column is `box cores busy`: 9–13.5 of 16 cores busy at 3–5 rps, while vLLM + agent +
driver together only accounted for 5–9 of them. `PROC_GROUPS` said the rest belonged to nobody.

---

## Iteration 1 — the 5 cores that belonged to nobody

**Saw.** Before firing anything, at complete idle: `load average 6.40`, and four orphaned Python
processes with `PPID 1`, started 21 Aug 14:12–14:34, elapsed 19h48m. A 3-second `/proc` delta
measured them burning **2.01 + 1.01 + 1.02 + 1.03 = 5.07 of 16 cores, with zero load offered**.
One was an `-m uvicorn ... --port 8001` parent that no longer answered on 8001; three were
`multiprocessing.spawn.spawn_main` children of it.

**Hypothesised.** These are leaked `uvicorn --workers` children from the `workers-4` and
`workers-2` runs. `stop_agent()` waited 30s, then `SIGKILL`ed only the uvicorn *parent* — which
orphans its children and reparents them to init, where nothing ever stops them. If so, then

1. every run from `workers-2` (14:40) onward was measured on a box that had already lost ~5 of its
   16 cores, so the whole table above understates the system, and
2. the unattributed cores are these: a `--workers` child is re-exec'd as
   `python3 -c from multiprocessing.spawn import spawn_main`, which contains neither `uvicorn` nor
   `driver.py`, so `PROC_GROUPS` filed it under nothing.

**Changed.** Two things, both in the harness rather than the system under test, because a
measurement I can't attribute is worth nothing:
`start_new_session=True` on the agent's `Popen` plus `os.killpg` in a single shared
`stop_agent()` (`scripts/phases/phase3.py` — `phase3` and `phase6` now route through one teardown),
and `spawn_main` added to the `agent` CPU group with a new `langfuse` group for the ClickHouse /
Postgres containers. Then killed the four orphans.

**Result.** Idle CPU **6.3 → 1.21 busy cores**; 5.1 cores back. The mechanism is confirmed by
construction, and the attribution hole is closed. Re-baselining at 10 rps on a clean box is the
next run — every number above is now a lower bound on the same configuration, so the baseline for
the rest of the phase has to be re-measured before any lever is judged.

**Re-baseline on the clean box, 10 rps for 300s.** The comparator is `query-budget`, not the
headline `workers-4`: it is the only 21 Aug run with the *same* `agent_workers=1`,
`max_iterations=3` and 10 rps, so the leak is the only variable. (An earlier draft of this table
paired `workers-4`'s latency with `workers-2`'s CPU figures and quoted a failure count belonging
to neither — corrected here, and the lesson is that a comparison across two runs is not one run.)

| | query-budget (contaminated) | clean-baseline | |
|---|---|---|---|
| p95 | 94.98s | **6.04s** | −94% |
| p50 | 1.86s | 1.30s | |
| failed requests | 1924 of 3000 | **0 of 3000** | |
| achieved rate | 8.33/s | **9.75/s** of 10 offered | |
| box cores busy | 9.53 of 16 | 5.81 of 16 | |
| vLLM p95 per call | 3.18s | 1.56s | |

So the entire collapse — 90+ second p95, 1900 failed requests, a permanent backlog — was **five
leaked cores**, not the agent design and not the serving config. Every conclusion drawn from the
21 Aug table about `--workers`, `MAX_ITERATIONS` or the query budget was drawn from noise; the
`workers-4` vs `workers-2` comparison in particular compared two runs that had a *different number
of leaked processes* running beside them.

Lesson worth more than the fix: the harness was measuring the box, not the system. `PROC_GROUPS`
was reporting 9 of 16 cores as belonging to nobody, and I read past that for four iterations
because the number I wanted was on a different row.

**Where it stands:** p95 6.04s against a 5s SLO at 9.75 rps, zero failures. A **1.04s miss** — a
real diagnosis problem instead of a collapse.

---

## Iteration 2 — the tail is the third attempt, not the serving layer

**Saw.** On the clean baseline the serving layer is comprehensively innocent: `queue_mean` 0.00s,
`prefill` 0.03s, `unaccounted` 0.01s, waiting peak **0**, preemptions **0/s**, KV at **4.1%** of
443K tokens, batch p95 364 of a 4096-token budget, and only **5.8 of 16 cores** busy. The GPU is
coasting and there is no queue anywhere in vLLM.

What the client latency distribution looks like instead (3000 runs):

| p50 | p75 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|
| 1.30s | 3.23s | 5.02s | 6.04s | 9.43s | 20.90s |

Smooth and heavy, not bimodal — no queueing signature, but 10.2% of runs over 5s. And
`calls_per_run` is **3.11**: measured, and almost exactly the Phase 5 mix (21 of 30 questions
single-shot = 2 calls, 3 revised once = 4 calls, 6 revised twice = 6 calls → mean 3.0).

Then the Langfuse breakdown of the 6 slowest runs (`--slowest 6 --label clean-baseline`) — every
single one had revised **twice**:

```
across the 6 slowest runs, by graph node:
     21.50s    46%  revise
     12.58s    27%  generate_sql
      6.34s    14%  verify
      6.26s    13%  execute
```

**Hypothesised.** The p95 is not a rate problem, it is an *arithmetic* problem: end-to-end latency
is the sum of sequentially dependent model calls, and a run that revises twice makes **six** of
them (generate, verify, revise, verify, revise, verify) at ~1.4–2.3s each. At `MAX_ITERATIONS = 3`
the ~20% of questions that revise twice are exactly the 10.2% of runs over 5s. Phase 5 measured
what that third attempt buys: per-iteration pass rate 33.3% → 36.7% → **36.7%**, i.e. *zero*
questions fixed at iteration 2 while 6 of 30 paid two extra calls for it. So dropping the cap
should take ~2 calls (~2.8–4.4s) off precisely the runs that form the tail, and cost nothing
measurable in quality.

**Changed.** `MAX_ITERATIONS` 3 → 2. One env var, nothing else.

**Result. SLO held.**

| | clean-baseline (`MAX_ITERATIONS=3`) | iters-2-clean (`=2`) | |
|---|---|---|---|
| **p95** | 6.04s | **3.78s** ✅ | **−2.26s** |
| p50 | 1.30s | 1.18s | −0.12s |
| p99 | 9.43s | 5.88s | −3.55s |
| max | 21.41s | 14.81s | −6.60s |
| achieved rate | 9.75/s | **9.88/s** | |
| failed requests | 0 | 0 | |
| **calls per run** | **3.11** | **2.63** | **−0.48** |
| vLLM p95 per call | 1.56s | 1.39s | −0.17s |
| decode mean | 0.63s | 0.54s | |
| box cores busy | 5.8 | 5.4 | |
| scrape success | 100% | 100% | |

**The targeted metric moved first, and end-to-end followed it.** `calls_per_run` 3.11 → 2.63 is the
mechanism: −0.48 calls per run × ~1.4s per call ≈ the −2.26s at p95, and the effect is
concentrated in the tail exactly as predicted (p50 barely moves at −0.12s, p99 drops −3.55s). The
serving layer got *slightly* faster too (vLLM p95 1.56 → 1.39s, decode 0.63 → 0.54s) purely because
there is less work in flight — 26.0 calls/s instead of 30.3.

Worth naming: **this is a quality lever spent as a latency lever.** Nothing about the serving config
changed. The 5s budget buys about three sequential model calls at this call latency, and the graph
was spending up to six. Whether it survived is a Phase 5 question, not a Grafana one — see the eval
below.

---
## Iteration 3 — push past the SLO: where does it actually break?

A green check at the SLO point says nothing about the margin, so: same configuration, 20 rps.

**Result: it breaks, and it breaks nowhere near the GPU.**

| | 10 rps (iters-2-clean) | 20 rps (push-20) |
|---|---|---|
| client p50 | 1.18s | **20.16s** |
| client p95 | 3.78s | **33.96s** |
| achieved rate | 9.88/s | 18.58/s of 20 offered — backlog |
| **agent overhead p50** | −0.08s | **18.52s (92% of p50)** |
| vLLM p50 / p95 per call | 0.48s / 1.39s | **0.62s / 1.72s** |
| calls/s served by vLLM | 26.0 | **49.2** |
| queue / waiting / preempt | 0.00s / 0 / 0 | **0.00s / 0 / 0** |
| KV used | 3.3% | **5.1%** |
| box cores busy | 5.4 of 16 | **7.0 of 16** |

vLLM doubled its throughput to 49 calls/s and barely noticed — per-call latency up 29%, zero
queueing, zero preemptions, KV at 5%. Meanwhile the client's p50 went up **17×** and **92% of it is
outside vLLM**. Nine of sixteen cores were idle. So the wall is inside the agent process, and it is
neither the GPU nor CPU exhaustion.

---

## Iteration 4 — the wall is 40 threads

**Saw.** `agent_overhead_p50` 18.52s with the box at 7 of 16 cores means requests are *waiting*
somewhere that has nothing to do with either resource. So I went looking for a fixed-size pool, and
measured the agent process directly during a 20 rps burst:

```
idle threads: 26
t=5s   threads=65   established=40
t=10s  threads=65   established=79
t=20s  threads=65   established=174
t=40s  threads=65   established=356
t=55s  threads=65   established=483
```

**Thread count pegs at 65 within the first five seconds and never moves again**, while accepted
connections climb without bound to 483. 65 = 26 idle threads + the 39/40 workers of anyio's default
thread limiter, which `anyio.to_thread.current_default_thread_limiter().total_tokens` confirms is
**40**.

**Hypothesised.** `/answer` was `def`, not `async def`. FastAPI runs a sync handler in anyio's
threadpool, so each in-flight agent run **owns one of 40 threads for its entire duration** — all
2.6 sequential model calls of it, every one of which is a network wait doing no work. That caps
concurrency at 40 runs regardless of how much GPU or CPU is free, and gives a throughput ceiling of
40 / (service time). At ~1.7s per run that is ~23 rps, and with GIL contention across 40 threads the
measured ceiling lands at ~15–18.6. Below the ceiling — 10 rps needs only ~13 concurrent — nothing
queues and the pool is invisible, which is why iterations 1–3 never saw it.

**Changed.** `async def answer` + `await graph.ainvoke(...)`, and the three model-calling nodes
(`generate_sql`, `verify`, `revise`) to `async def` + `await llm().ainvoke(...)`. `execute_node`
stays **sync** deliberately: sqlite is genuinely blocking work, and LangGraph runs a sync node in a
worker thread when the graph is driven with `ainvoke` — so the blocking part borrows a thread for
the milliseconds it needs, instead of one thread being held for the whole run. Nothing else moved:
same prompts, same graph shape, same `MAX_ITERATIONS=2`, same vLLM flags.

**Result: the ceiling moved, and the SLO now holds at twice the required rate.** Same 20 rps,
same everything else:

| | push-20 (sync handler) | async-20 | |
|---|---|---|---|
| **client p95** | 33.96s | **4.95s** ✅ | **−29.0s** |
| client p50 | 20.16s | **1.60s** | −18.6s |
| client p99 | 36.81s | 7.86s | |
| **agent overhead p50** | **18.52s (92%)** | **−0.08s (0%)** | the whole win |
| achieved rate | 18.58/s (backlog) | **19.65/s of 20** | |
| failed requests | 0 of 6000 | 0 of 6000 | |
| vLLM p50 / p95 per call | 0.62s / 1.72s | 0.63s / 1.76s | **unchanged** |
| calls/s served | 49.2 | 52.0 | |
| queue / waiting / preempt | 0.00s / 0 / 0 | 0.00s / 0 / 0 | unchanged |
| KV used | 5.1% | 5.4% | unchanged |
| box cores busy | 7.0 of 16 | 7.3 of 16 | unchanged |

**The targeted metric moved and took the SLO with it.** `agent_overhead_p50` 18.52s → −0.08s: the
18.5 seconds were *entirely* time spent waiting for one of 40 threads, and they vanished. Every
vLLM-side number is unchanged to within noise, which is the confirmation — the change touched
nothing about serving and the serving layer noticed nothing.

**And the honest part: at the SLO point itself, this change made things slightly worse.**

| 10 rps | iters-2-clean (sync) | async-10 |
|---|---|---|
| p95 | **3.78s** | 4.09s |
| p50 | 1.18s | 1.20s |
| calls/run | 2.63 | 2.61 |

At 10 rps only ~13 runs are ever in flight, so the 40-thread pool never binds and there is nothing
for the fix to fix — what's left is that the event loop serialises all the Python-level work
(message construction, JSON, Langfuse span building) onto one thread where 40 OS threads were
previously interleaved. +0.31s at p95, at or just outside run-to-run noise. This is the
"a metric improved and the SLO didn't follow" lesson in miniature: the change is worth keeping not
because it made the SLO point faster, but because it moved the **capacity ceiling** from ~18 rps to
past 20, and capacity is what a 10 rps SLO needs headroom in.

---

## Iteration 5 (diagnosis only, no change) — where it breaks now, and why I stopped

Pushed to 30 rps. It breaks, and for the **first time in the entire phase the signature is
different**:

| | async-20 (pass) | async-30 (collapse) |
|---|---|---|
| client p50 / p95 | 1.60s / 4.95s | 45.66s / 108.78s |
| **vLLM waiting peak** | **0** | **245** ← first non-zero all phase |
| **vLLM queue mean** | **0.00s** | **0.34s** |
| vLLM p50 / p95 per call | 0.63s / 1.76s | 1.35s / 4.56s |
| decode mean | 0.70s | 1.19s |
| KV used | 5.4% | 12.2% |
| **calls/s served** | **52.0** | **42.3** ← *down* |
| failed requests | 0 of 6000 | **5739 of 9000** |
| box cores busy | 7.3 of 16 | 7.0 of 16 |

Two things at once. vLLM finally has a real queue — `num_requests_waiting` peaked at 245 and
`queue_time` became nonzero, having been exactly 0.00s in all twelve prior runs. And the system
*lost* throughput going from 20 → 30 rps offered: 52 calls/s down to 42. That is congestion
collapse, not saturation. The mechanism, in order:

1. Removing the 40-thread pool also removed the only thing that was doing **admission control**.
   The async handler now accepts unbounded concurrent runs.
2. Each accepted run immediately issues its model call, so vLLM sees thousands of concurrent
   requests and its scheduler queue grows to 245.
3. Per-call latency triples, so runs exceed the driver's 120s timeout and the client disconnects.
4. vLLM keeps decoding for clients that are already gone — 5739 requests' worth of wasted GPU —
   which is why *served* throughput falls while offered load rises.

**So the capacity of this stack is ~20–21 agent runs/s**, against a required 10. Past capacity it
queues without bound rather than degrading gracefully.

**I stopped here rather than fixing it, deliberately.** The SLO is 10 rps and the system holds it at
double that with zero failures; bounded concurrency plus load shedding is a real improvement but it
is work the SLO does not ask for, and I would only be able to half-validate it (a semaphore alone
converts "everything collapses" into "everything queues" — the client still times out; doing it
properly means returning 503 above a concurrency watermark, which is a design decision, not a
tuning knob). It is named in `REPORT.md` under what I'd do with more time, with the measured
capacity number to size it against.

---

## Did quality survive? (`results/eval_after_tuning.json`)

`MAX_ITERATIONS` 3 → 2 was a *quality* lever spent for latency, so the eval is the only thing that
can say whether it was free. Same 30 questions, same H100, same prompts:

| | baseline (`results/eval_baseline.json`) | after tuning |
|---|---|---|
| pass rate (multiset, headline) | 36.67% (11/30) | **36.67% (11/30)** |
| pass rate (set-wise, BIRD's own) | 40.0% | **40.0%** |
| per-iteration pass rate | 33.3 → 36.7 → 36.7 | **33.3 → 36.7** |
| `loop_delta` | +3.3% | **+3.3%** |
| verifier `rejected_correct` | 0 | **0** |
| verifier `accepted_wrong` | 13 | 12 |
| questions that revised | 9 | 10 |
| mean latency | 1.07s | **0.79s** (−26%) |
| p95 latency | 2.41s | **1.58s** (−34%) |
| max latency | 3.10s | **1.73s** (−44%) |

**Identical pass rate, identical loop contribution, 26–34% less latency.** Phase 5's prediction was
exact: the third attempt fixed zero questions, so removing it cost zero questions. The loop still
earns its +3.3% at iteration 1, and it still never damages a correct answer (`rejected_correct = 0`).

---

## Verdict

**SLO hit, with 2× margin.**

| | required | measured (final config, 300s) |
|---|---|---|
| p95 end-to-end agent latency | < 5s | **4.09s** at 10 rps · **4.95s** at 20 rps |
| rate | ≥ 10 runs/s | **9.92/s** offered 10 · **19.65/s** offered 20 |
| window | 5 min | 300s |
| failed requests | — | **0 of 3000** · **0 of 6000** |
| scrape success | ≥95% (mine) | 100% |
| quality | must not regress | pass rate **unchanged** at 36.67% |

Measured capacity ceiling: **~20–21 runs/s**. Above it the system collapses rather than degrading.

Final configuration: vLLM flags unchanged throughout Phase 6 — FP8, prefix caching, async
scheduling, `--max-num-seqs 256`, and **`--api-server-count 8`**. That 8 is worth flagging: the
launch script says 4, but the running cmdline carries `--api-server-count 4 --api-server-count 8`
from an earlier `restart_vllm.sh` override, and last flag wins. `census()` reported 8 on every run
in this phase and it took cross-checking that against the script to notice. Untested at 4 vs 8 here
and almost certainly irrelevant — the front end never queued (`unaccounted` 0.01s, 100% scrape
success, 12ms scrape duration), which is the only thing the flag treats.

Agent side: `MAX_ITERATIONS=2`, one uvicorn process, async handler and async LLM nodes. The
committed **default is 3** - the brief's range, and what produced `eval_baseline.json` - so every
run from iteration 2 on was driven with `MAX_ITERATIONS=2` in the environment.

**Everything Phase 6 changed was above the serving layer.** Not one vLLM flag moved. Across all
thirteen runs, `queue_time` was 0.00s and preemptions were 0/s until the box was pushed to 3× the
SLO, KV never exceeded 12% of 443K tokens, and across the six clean runs no run used more than 7.3
of 16 cores (the contaminated runs burned 9.5-13.5, which is iteration 1's evidence). The
Phase 1 configuration was never the constraint; a leaked process group, an iteration cap and a
40-thread pool were.

---

## Grafana windows (for the screenshots)

Prometheus retention still covers the 21 Aug runs, so every window below is capturable now.

| label | rps | p95 | window |
|---|---|---|---|
| workers-2 (contaminated) | 10 | 111.80s | `from=1787322874995&to=1787323250000` |
| clean-baseline | 10 | 6.04s | `from=1787393002295&to=1787393325000` |
| iters-2-clean | 10 | 3.78s | `from=1787393447614&to=1787393751411` |
| **push-20** *(before)* | 20 | 33.96s | `from=1787393780948&to=1787394104064` |
| **async-20** *(after)* | 20 | **4.95s** | `from=1787394447063&to=1787394752614` |
| async-10 *(final, SLO point)* | 10 | **4.09s** | `from=1787394771781&to=1787395074464` |
| async-30 *(the new cliff)* | 30 | 108.78s | `from=1787395093757&to=1787395454173` |

Prefix each with `http://localhost:3000/d/vllm-serving?`.

`screenshots/grafana_before.png` / `grafana_after.png` are **push-20 → async-20**: the pair around
the change that moved the needle, at identical offered load. Both are captured, and side by side
they are the most useful artefact in the phase — because **they are almost indistinguishable.**
Read off the two dashboards:

| Panel | before (client p95 **33.96s**, FAIL) | after (client p95 **4.95s**, PASS) |
|---|---|---|
| vLLM call p95 | 1.7s | 1.8s |
| **"Calls under 5s"** | **99.7%** | **99.6%** |
| Throughput | 49.2 req/s | 52.0 req/s |
| waiting | 0 | 0 |
| KV cache | 4.0% | 4.4% |
| Preemptions/s | 0 | 0 |
| e2e latency (p50/p95/p99) | ~0.6 / ~2 / ~4s | ~0.6 / ~2 / ~4s |
| where the time goes | decode ~700ms | decode ~700ms |
| TTFT p99 | 50–150ms | 40–100ms |
| inter-token latency | ~17–25ms | ~17–25ms |
| batch fullness | ~1K of 4K | ~1K of 4K |
| prefix hit rate | ~90% | ~90% |

Every panel is flat across a change that moved end-to-end p95 by **29 seconds**. The sharpest
detail is the stat tile: the serving dashboard's own "calls under 5s" reads **99.7% during the run
that missed the SLO by 29s**, and 99.6% — marginally *worse* — during the run that passed. If you
only had this dashboard, you would conclude nothing was wrong.

That is not a broken dashboard, it is a correct one being asked the wrong question. It monitors one
vLLM call; the SLO is on an agent *run*, which is 2.6 of those calls plus everything the agent does
between them. The 18.5s was spent queueing for a thread before a call was ever issued, so it was
invisible here by construction — and the only reason it got found is that
`agent_overhead_p50` compares the two layers instead of trusting either alone.

**The complement to the flat pair** — the one place in the phase where the vLLM panels *do* move —
is async-20 → async-30 (`from=1787395093757&to=1787395454173`): waiting 0 → **245**, queue
0.00 → 0.34s, KV 4.4% → 12.2%, per-call p95 1.8 → 4.6s. Worth a capture if the report has room; it
is what "vLLM is finally the constraint" actually looks like, and it took 3× the SLO to produce it.
