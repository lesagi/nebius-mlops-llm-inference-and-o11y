# Text-to-SQL on Qwen3-30B-A3B — report

Qwen3-30B-A3B on one H100 under vLLM, with a LangGraph `verify → revise` agent on top, traced in
Langfuse and measured against BIRD. Detail lives outside this file by design: **`PHASE6_LOG.md`**
(the full SLO log, hypothesis before result), `results/` (`phase1_sweep`, `eval_baseline`,
`eval_schema_values`, `eval_after_tuning`, `phase6_slo`), one runner per phase from 2 on in `scripts/phases/`.

## 0. Running this

**One deviation from README Phase 0, step 2.** It says `cp .env.example .env`; no template ships,
because the only `.env` this project ever had holds real credentials and is gitignored. The brief
is left exactly as issued rather than edited to match, so: create `.env` in the repo root by hand.
Every variable has a working default except the Langfuse pair, so the minimum for a full run is:

```bash
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8   # the checkpoint, unaliased (§1)
VLLM_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=not-needed                          # vLLM ignores it; a hosted provider would not
LANGFUSE_PUBLIC_KEY=pk-lf-...                      # omit both to run with tracing off
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
HF_TOKEN=hf_...                                    # only to download the checkpoint
```

Three optional knobs, all read at agent start-up and reported by `GET /health`:
`MAX_ITERATIONS` (default 3, §3), `SCHEMA_SAMPLE_VALUES` (default 0, §2) and
`QUERY_BUDGET_SECONDS` (default 2.0). Otherwise follow README Phase 0 — `uv sync`,
`uv run python scripts/load_data.py`, `docker compose up -d` — then
`./scripts/start_vllm.sh` and `uvicorn agent.server:app --port 8001`.

## 1. Serving configuration (Phase 1)

`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, vLLM 0.10.2 (V1), launched by `./scripts/start_vllm.sh`,
which passes extra args through to `vllm serve` so one lever moves at a time. The checkpoint id is
unaliased, so the agent's request, the Langfuse trace and vLLM's `model_name` label all name the
quantization that actually served.

**Workload measured before choosing anything:** 11 BIRD schemas, only **8.6K tokens in total**; real
prompts 518–1211 tokens; outputs ~72 tokens; 2–3 *dependent* calls per agent run. A tiny reusable
prompt corpus, short outputs, decode-dominated.

| Flag | Value | Why |
|---|---|---|
| model | `...-2507-FP8` | Block-wise FP8 (128×128, e4m3) halves weights 61→29 GiB, taking KV from ~12 to **41 GiB**; better quality than on-the-fly `--quantization fp8`. |
| `--max-model-len` | 8192 | Largest real prompt ~2.6K; 8192 leaves 3× headroom and caps a runaway request, vs the native 262144. |
| `--gpu-memory-utilization` | 0.90 | 41 GiB KV = 443K tokens. Peak measured use across the whole project: 12%. |
| `--max-num-seqs` | 256 | Measured, not reasoned: 64 → 256 cut p95 5.13 → 3.63s, ITL 17.9 → 12.6ms. |
| `--max-num-batched-tokens` | 4096 | 2048/4096/8192 indistinguishable — prefix caching leaves almost no prefill to schedule. |
| `--enable-prefix-caching` | on | All schemas fit in cache and precede the question in every prompt → 93.6% of prompt tokens hit; worth ~2×. |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution; A3B activates 3B params, so short steps make per-step CPU a real share of ITL. A/B: p95 4.13 → 3.68s. |
| `--api-server-count` | 8 | One process couldn't keep up: 87% of e2e sat between front end and scheduler, `/metrics` took 17s+, blinding Prometheus. 4 fixed it; the 8 arrived as drift, which the config census caught (§3.3), and `start_vllm.sh` now passes 8 so it reproduces the runs below. |
| `--disable-log-requests` | on | Costs CPU at 25+ req/s, says nothing `/metrics` doesn't. |
| `--tensor-parallel-size` | 1 (default, not passed) | One GPU — which also rules out expert parallelism. |
| `--kv-cache-dtype` | **off** | Would halve KV bytes/token, but KV peaked at 12% of 41 GiB — buying more of the most abundant resource at an accuracy cost. |
| speculative decoding | **rejected** | Looked ideal (SQL copies schema identifiers verbatim), measured *worse*: p50 2.82 → 3.40s, and 0.10.2 won't combine it with `--async-scheduling`. |

Those rows come from `load_test/vllm_probe.py` replaying calls over 60s — the serving layer, not
agent runs over the SLO's 5 minutes, so §3 supersedes them. Two findings outlive them.

**Prefix caching pays out in ITL, not TTFT.** Turning it off moved TTFT only 109→169ms while ITL
nearly doubled: with chunked prefill, uncached prefill chunks share each engine step with in-flight
decodes, so the cost lands on *every* concurrent request rather than on the one that caused it.
That is why Phase 3's prompts put the schema ahead of the question — it keeps the reusable half of
the prompt cacheable.

**The `--max-num-seqs` win is unexplained, and I am reporting it that way.** Raising a cap that was
never reached should do nothing, yet it cut p95 30%. `/metrics` rules out every routine cause:
`num_requests_waiting` 0.0, zero preemptions, KV at 4.7%. Reproduced 3× in both orders, so it is
not ordering noise. The config takes the win and the mechanism is still open.

**FP8 was controlled against bf16, and costs nothing measurable** (§2.1). That closes the one
open risk this section used to carry.

One deviation from the brief worth naming: `README.md`'s snippet uses
`from langfuse.callback import CallbackHandler`, the pre-v3 path, which raises
`ModuleNotFoundError` against the pinned `langfuse==4.7.1`. `agent/server.py` uses
`from langfuse.langchain import CallbackHandler` instead.

## 2. Baseline eval (Phase 5)

30 BIRD questions, execution accuracy over canonicalized row sets, **every attempt scored** rather
than only the last (`results/eval_baseline.json`).

| | |
|---|---|
| pass rate (multiset, headline) | **36.7%** (11/30) |
| pass rate (set-wise, BIRD's own) | **40.0%** — one DISTINCT-class disagreement |
| per-iteration | 33.3 → 36.7 → **36.7** |
| iterations taken | 21 single-shot, 3 revised once, 6 revised twice |
| verifier verdicts | accepted 11 correct / **13 wrong**; rejected 21 wrong / **0 correct** |

Both rates are reported because the provided `canonicalize` compares row *multisets*, so a correct
query missing `DISTINCT` fails it while BIRD's set-wise eval passes it — and set-wise is *looser*,
not better, since gold Q30 legitimately returns duplicate rows a set comparison would let a
prediction collapse. Of the 19 failures, **11 were never challenged**: the verifier accepted them at
iteration 0, so revise never ran. Eight were revised and still wrong, five returning zero rows every
time on near-identical SQL.

**One experiment, run and rejected.** Every revise-triggering failure was a literal the schema never
showed (`molecule.label = 'carcinogenic'` where values are `'+'`/`'-'`; `'Cl'` where the data is
`'cl'`), so `render_schema` was changed to append distinct values of low-cardinality text columns.
Pass rate **fell to 30.0%** (`eval_schema_values.json`). Why matters more than the change: *partial*
annotation biased column choice — Q5 was right on unannotated `district.A2` and wrong on annotated
`district.A3`, under greedy decoding, so the schema text was the only variable; values gave
vocabulary but not semantics (Q25 chose `Admission = '+'`, gold wants `'-'`); and it **blinded the
verifier**, whose best signal was zero-rows-and-NULL — `accepted_wrong` rose 13 → 17. Off by default,
kept behind `SCHEMA_SAMPLE_VALUES=1`: the negative result is the evidence.

### 2.1 The bf16 control — is 36.7% a quantization artifact?

Serving `...-2507` unquantized, same prompts, same cap of 3, same 30 questions
(`results/eval_bf16_control.json`):

| | FP8 | bf16 |
|---|---|---|
| pass rate (multiset / set-wise) | **36.7% / 40.0%** | 30.0% / 33.3% |
| **iteration-0 pass rate (greedy, deterministic)** | **33.3%** | **30.0%** |
| `loop_delta` | +3.3% | +0.0% |
| `accepted_wrong` | 13 | 13 |
| mean latency | 1.07s | 1.03s |

**bf16 scored *lower*, and the honest reading is that the two are indistinguishable.** Only 2 of 30
questions changed verdict, and they are different animals. `generate_sql` decodes greedily, so its
output is deterministic per model: **14 of 30 first attempts differ textually between the two
models, but only 1 of those changes the verdict** — "which active district has the highest average
score in Reading", where bf16 read it as `AVG(AvgScrRead) GROUP BY District` (Santa Cruz County
Office of Education) and FP8 read `AvgScrRead` as the already-averaged column and sorted it
(Palo Alto Unified, which is gold). Both are defensible English; gold agrees with FP8. The other
flip is `card_games` Q18, where the iteration-0 SQL is *byte-identical* across models and the
divergence happened inside the revise, which samples at 0.7 — that is noise, not quantization.

So the −6.7 point gap is one deterministic question (−3.3, quantization) plus one sampled one
(−3.3, noise). At n=30 a single question is 3.3 points. **Quantization is not what caps this system
at 36.7%** — `accepted_wrong` is 13 under both models, which points at the verifier (§5), not the
weights. FP8 keeps its place on the memory argument it was chosen for, not a quality one: at a fixed
`--gpu-memory-utilization 0.90` the ~32 GiB that bf16 weights reclaim comes straight out of KV,
off the 40.60 GiB / 443,456 tokens measured under FP8. (I did not capture the bf16 startup
banner — it went to an interactive terminal — and `cache_config_info` reports
`num_gpu_blocks: None`, so treat that as arithmetic rather than a second measurement.)

## 3. Hitting the SLO (Phase 6)

> **P95 end-to-end agent latency under 5s, 10+ agent runs/s, over a 5-minute window.**

**Verdict: hit, with 2× margin.** p95 **4.09s at 9.92 runs/s**, same config **4.95s at 19.65
runs/s** — 300s each, zero failures of 3000 and 6000 requests. Pass means four things: p95 < 5s,
zero failures, achieved rate ≥95% of offered (a shortfall *is* a backlog), and ≥95% scrape success
(a window with gaps is not evidence).

**Baseline against the SLO:** p95 **6.04s** at 9.75 runs/s, zero failures — a **1.04s miss**,
measured only after iteration 1 made the box trustworthy.

| # | Saw | Hypothesised → changed | Result |
|---|---|---|---|
| 1 | 5.07 of 16 cores busy at **zero load**; four 19-hour-old `PPID 1` Python processes | leaked `uvicorn --workers` children — `stop_agent` killed only the parent, and a worker re-execs as `spawn_main`, so the CPU census never saw them → `start_new_session` + `killpg` | idle CPU 6.3 → **1.21** cores; re-baseline against `query-budget`, the run with the same workers/iterations/rate: p95 **94.98 → 6.04s**, failures **1924 → 0** of 3000, cores **9.53 → 5.81**. The "collapse" was five stolen cores |
| 2 | vLLM innocent (queue 0.00s, waiting 0, KV 4.1%); Langfuse: all 6 slowest runs revised twice, `revise` = 46% of tail | latency is a sum of *sequentially dependent* calls, and Phase 5 showed iteration 2 fixes zero questions while 6/30 pay → `MAX_ITERATIONS` 3 → **2** | calls/run 3.11 → 2.63, p95 **6.04 → 3.78s** ✅, p50 flat — tail-only, as predicted |
| 3 | at 20 rps: p50 20.2s, `agent_overhead_p50` **18.5s = 92%**, yet vLLM 0.62s/call and **9 of 16 cores idle** | a fixed-size pool, not a resource limit → measured the process directly | threads **pegged at 65 within 5s** and never moved (26 idle + 39 of anyio's **40**-token limiter) while accepted connections climbed 40 → 483 |
| 4 | the above | `/answer` was `def`, so FastAPI ran it in the threadpool: every run owned one of 40 threads for all 2.6 of its sequential *network waits*, a ceiling of ~18 rps invisible at 10 → `async def` + `graph.ainvoke`; `execute_node` stays sync, sqlite being real blocking work | overhead 18.52 → **−0.08s**, p95 **33.96 → 4.95s** ✅ at 19.65 rps, **every vLLM number unchanged** |
| 5 | at 30 rps: `waiting` peaks at **245**, first non-zero all phase, and served throughput *falls* 52 → 42 calls/s while 5739 of 9000 fail | async also removed the only admission control: unbounded accepts → queue → calls outlive the client timeout → the GPU decodes for clients already gone → left deliberately (§5) | capacity **~20–21 runs/s**. Past it the system queues without bound instead of degrading |

On the cap: the brief says 3-5, so `MAX_ITERATIONS` **ships at 3**, which also reproduces
`eval_baseline.json`. The SLO figures above are at **2**, set by env var — the one lever this
phase spent, and §3.2 shows it cost no quality.

Async made the SLO *point* marginally **worse** (p95 3.78 → 4.09s at 10 rps), where only ~13 runs are
in flight and the pool never binds; it earns its place on capacity, not latency. A metric improving
without the SLO following is a lesson the brief warns about — this is its mirror image.

### 3.1 The before/after pair looks like nothing happened

`grafana_before.png` / `grafana_after.png` bracket iteration 4: identical load, client p95 **33.96s
vs 4.95s**, and the panels are indistinguishable — call p95 1.7 → 1.8s, throughput 49.2 → 52.0
req/s, waiting 0, KV 4.0 → 4.4%. The "calls under 5s" tile reads **99.7% during the run that missed
the SLO by 29 seconds**, and 99.6% during the run that passed.

The dashboard isn't broken; it measures one vLLM call while the SLO is on a *run* — 2.6 calls plus
the queueing between them, and the missing 18.5s was spent waiting for a thread *before* any call
was issued. What found it was `agent_overhead_p50`, client p50 minus (calls/run × vLLM p50): the
one number belonging to neither layer. **Grafana proved the serving layer innocent, Langfuse
localised the rest inside the run, and the subtraction between them found what neither showed.**

### 3.2 Quality survived

| | baseline | after tuning |
|---|---|---|
| pass rate (multiset / set-wise) | 36.67% / 40.0% | **36.67% / 40.0%** |
| per-iteration | 33.3 → 36.7 → 36.7 | **33.3 → 36.7** |
| verifier `rejected_correct` | 0 | **0** |
| eval mean / p95 latency | 1.07 / 2.41s | **0.79 / 1.58s** |

Reproduce: the cap belongs to the **agent** process, not the eval — `run_eval.py` only POSTs to
`:8001`. So start the agent under the cap you want and check `/health`, which reports it:

```bash
uvicorn agent.server:app --port 8001                  # default 3  -> eval_baseline.json
MAX_ITERATIONS=2 uvicorn agent.server:app --port 8001  # tuned 2   -> eval_after_tuning.json
uv run python evals/run_eval.py --out results/<file>.json
```

**Zero quality cost, ~30% less latency.** Phase 5's prediction was exact: the third attempt fixed
zero questions, so removing it cost zero questions.

### 3.3 What Phase 6 says about Phase 1

**Not one vLLM flag moved.** One had already moved and the census caught it: every run of the clean
re-measurement was served by `--api-server-count 8`, not the 4 in `start_vllm.sh`, because an earlier `restart_vllm.sh
--api-server-count 8` appended a second value and last flag wins. Cross-checking the census against
the script — rather than trusting the script — is why it surfaced.

Across 13 runs `queue_time` stayed at 0.00s and preemptions at 0/s until 3× the SLO, KV never
passed 12% of 443K tokens, and batch occupancy never passed ~950 of 4096; across the six *clean*
runs, no run used more than 7.3 of 16 cores. **The serving config was never the constraint.** What
was: a leaked process group, an iteration cap the eval had already shown worthless, and a 40-thread
default in the web framework.

## 4. Did the agent loop earn its keep?

Yes, by a smaller margin than the architecture suggests. It is worth **+3.3%** — 33.3% at iteration
0 to 36.7% — which is one question of thirty: `card_games` Q18, where the first attempt returned four
rows with inconsistent NULLs, the verifier said exactly that, and the revise added `DISTINCT` and
`IS NOT NULL` and was correct. Against that it has never damaged a correct answer across two full
evals (`rejected_correct` = 0 both times), so it is cheap insurance whose cost is latency, not
quality — 2.6 model calls per run instead of 1, with the third iteration removable for free. The
ceiling is not the loop but the **verifier**: it accepted 12–13 wrong answers, so 11 of 19 failures
were never challenged, because a defensible query over defensible-looking rows is invisible to a
verifier that sees only the question, the SQL and the rows. The bf16 control sharpens this: that one question is decided inside a
temperature-0.7 revise, and bf16 lost it, taking `loop_delta` to exactly 0.0%. So the loop's
measured value rests on a single *sampled* question — real, but not something n=30 can call
reliably. Phase 6 added one win for free — the 2s
query budget's cancellation message is phrased for the verifier, so the loop now repairs a failure
mode it was never shown in Phase 3.

## 5. With more time

- **Admission control, sized against the measured 20–21 runs/s.** Return 503 above a concurrency
  watermark. Going async removed the last implicit backpressure, and at 30 rps that produces
  congestion collapse — throughput *falling* while the GPU decodes answers for clients that already
  timed out. Shedding is the only response that stops a queue outliving its client.
- **Attack the verifier, not the loop** — it is the measured ceiling on quality. It currently sees
  the question, the SQL and the rows but **not the schema**, so it cannot notice a wrong column.
  Cheapest first: give `VERIFY_USER` the schema; then a second opinion at temperature requiring
  agreement. Track `accepted_wrong`, not the pass rate — that is the number that has to fall.
- **BIRD's `value_description` CSVs, annotating every column or none.** §2 says values were the wrong
  half: the model needed to know `A15` means "crimes 1995" and `'-'` means outpatient — semantics,
  not vocabulary. The CSVs carry that for all 9 eval DBs, and the failed experiment's lesson is that
  partial annotation is itself a bias.
- **Sample tracing under load.** Tracing every request is right at 10 rps and probably wrong at 20;
  its cost was never isolated in a run, which is the reason to measure it rather than assume.
- **Repeat the bf16 control with a fixed revise seed.** §2.1 answered the quantization question, but
  half its gap sat in a temperature-0.7 revise. Pinning the seed would separate model from sampling
  properly instead of reasoning about which flips are deterministic.
- **Bound the eval harness's own SQL.** The agent caps a query at 2s; `run_eval.py` re-runs both
  SQLs unbounded (`sqlite3.connect(timeout=)` bounds lock acquisition, not runtime), so an attempt
  the agent aborted could be scored correct. Measured impact on both evals is **zero**
  (`sql_error_rate` 0.0), but it is a real hole in the scorer.
- **Not worth doing:** `--kv-cache-dtype fp8` (KV peaked at 12%) or more `--api-server-count` tuning
  (the front end never queued once the leak was gone).
