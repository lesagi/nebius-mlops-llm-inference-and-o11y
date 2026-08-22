# Home assignment: LLM inference + o11y

  

## Legend

Imagine you're responsible for the LLM part of an internal analytics product (aka talk-to-your-data-for-those-with-sensitive-data). You need to show a text-to-SQL PoC for a trivial workflow:
1. analysts ask questions in English
2. the system runs SQL against an internal warehouse and returns rows. 

You're to present it to technical leadership and they'll buy it if they see the whole system running with [Qwen3-30B-A3B](https://huggingface.co/collections/Qwen/qwen3) against [BIRD-bench](https://bird-bench.github.io/) showing some decent quality and performance.

All in all you need to do two things:
- Setup inference infra (aka vLLM and Prometheus + Grafana are your friends). 
- Put an agent on top of that infra (aka LangGraph and Langfuse are your friends). The purpose of agent is to boost quality of system's responses.
  
The endpoint will run on one H100, ain't much but honest hardware.

![Farmer](https://cloudfront-us-east-1.images.arcpublishing.com/gray/FLBGRRRDQNHYBNTNHU4WOWRIFY.png)

Disclaimer:
You don't need to set up Kubernetes, build a frontend or productionize past the toy-infra level. The point of the whole assignment is to learn what each layer tells you.

---

## Answering your why shall I even cares

By the end of the assignment you should be able to:

- Deploy an open-source MoE LLM with vLLM and make an informed inference config decision.
- Read vLLM's `/metrics` and build a Grafana dashboard that tells you what your serving layer is doing right now.
- Build a multi-step agent in LangGraph where the architecture itself adds measurable value.
- Use Langfuse to inspect agent traces and explain why one request was slow when another wasn't.
- Build an offline eval system that separates capability regression from latency improvement.
- Combine both observability layers to find and fix a bottleneck, then prove you fixed it without breaking quality.

If done with curiosity, this assignment will bring you tons of knowledge about how things work which is invaluable. 

---

## Prerequisites

  

-  **Hardware:** 1× H100

-  **Software:** Docker + docker-compose, Python with `python3-dev` headers (vLLM's torch.compile path needs them), uv, git  

---


## Phase 0 (Setup)

  
You'll be working on a cloud VM. All services in this assignment listen on `localhost` on the VM, so to reach their UIs from your laptop browser you need to forward ports over your SSH session.

You need five ports forwarded for the full assignment: **3000** (Grafana), **9090** (Prometheus), **3001** (Langfuse), **8000** (vLLM), **8001** (your agent server).


**VSCode or Cursor.** Both have a Remote-SSH extension - install it, then `F1` → *Remote-SSH: Connect to Host* → add the VM. Once connected, the *Ports* panel at the bottom of the editor lets you forward each port with one click: hit *Forward a Port*, type `3000`, repeat for the other four. You also get a local-feeling editor working on the remote files, which makes the rest of the assignment much less painful.

  

**Plain SSH (fallback).**

```bash
ssh -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    <user>@<vm-host>
```

  

Each `-L <local-port>:localhost:<vm-port>` line forwards a port on your local machine to the same port on the VM. With the session open, `http://localhost:3000` in your local browser hits Grafana on the VM.

Once connected and forwarded, run the rest on the VM:

```bash
# 1. Clone repo and install dependencies
git clone <repo-url>
cd <repo-folder>
uv sync

# 2. Configure environment (Langfuse keys go here in Phase 4)
cp .env.example .env

# 3. Load BIRD subset (~500 MB sqlite + JSONs)
uv run python scripts/load_data.py

# 4. Start the o11y stack
docker compose up -d
```

Sanity-check from your laptop browser:

  

- Prometheus → http://localhost:9090

- Grafana → http://localhost:3000 (admin / admin)

- Langfuse → http://localhost:3001 (sign up locally, instant)

  

If a URL doesn't load, the port forward is the most likely culprit.

### What you should have in the end:
- Five ports forwarded; three UIs reachable from your laptop browser
- BIRD data under `data/bird/`
- `.env` created from the template


---

  

## Phase 1 (vLLM)

Imagine the minimal SLO your leadership can buy is something like this:

> **P95 end-to-end agent latency under 5 seconds, 10+ RPS (1rps = 1 full agent run per second) over a 5-minute window.**


The model is fixed: `Qwen/Qwen3-30B-A3B-Instruct-2507`. The hardware is fixed: 1× H100 80GB. Everything else is up to you, use your knowledge of inference optimizations.

We are not enumerating which parameters to consider on purpose. Knowing which levers to reach for, given a workload profile (1.5-3K-token prompts, short structured outputs, ~2-3 dependent calls per user request) and a latency target, is the apply-the-lectures part of the assignment. Heads-up: you'll need to iterate.   

There's an example launch script at `scripts/start_vllm.sh` to get you started - feel free to modify it or roll your own. The [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) are your reference for the available flags.

### What to do:
1.  Start vLLM with your initial configuration.

2.  Confirm the model loads, responds, and the output looks reasonable. Try firing 3-5 inputs from `evals/eval_set.jsonl` manually.
3. Find your config. You'll probably need to revisit this once you have your agent running.
4.  Write down your configuration in `REPORT.md` and explain it.

### What you should have in the end:
- vLLM serving Qwen3-30B-A3B at `http://localhost:8000`
- A few manual queries returning sensible SQL
- A screenshot of vLLM serving + one manual query returning SQL (`screenshots/vllm_manual_query.png`)
- Your config flags + one-line justifications in `REPORT.md`

---

  

## H100 is not needed all the time

You don't have to occupy an H100 VM to make progress on every phase. The agent and the o11y stack talk to *any* OpenAI-compatible server, so you can build and debug against a lighter backend and switch to the real endpoint only when the numbers matter. Configure the backend via `VLLM_BASE_URL` / `VLLM_MODEL` / `OPENAI_API_KEY` in `.env` (see the commented block there). Consider two options:

- Hosted API: point at e.g. OpenAI with a your own key. It exposes no Prometheus metrics though.
- CPU-only vLLM: run vLLM on CPU with a small stand-in model like `Qwen/Qwen3-0.6B`. See the [CPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html) for more details.

What you can do off the H100:

| Phase | Off-GPU? | Notes |
|---|---|---|
| 2 (Grafana) | CPU-vLLM only | Hosted APIs expose no `/metrics`. Build panels and confirm they react against a CPU vLLM; absolute numbers are unrepresentative. |
| 3 (Agent) | Either | Pure graph / prompt wiring. |
| 4 (Tracing) | Either | Langfuse captures the LangGraph spans regardless of backend. |
| 5 (Evals) | Either | Validate the eval harness end-to-end; real pass rates must come from the 30B endpoint. |

Anything you report e.g. eval pass rates, latency, the Phase 6 SLO must come from the real `Qwen3-30B-A3B` on the H100.

---

## Phase 2 (o11y core)

Prometheus is already configured to scrape vLLM's `/metrics`. Grafana is already configured with Prometheus as a datasource and a starter dashboard with 2 pre-built panels.

### What to do:

Open the starter dashboard in Grafana and build it out to cover three categories of serving health, drawing from what vLLM exposes at `/metrics`:

1.  **Latency**, with percentiles. The dashboard should let you look at it during a load test and answer "is the system slow, and if so, where in the request lifecycle?"
2.  **Throughput**, with percentiles where it makes sense. Tokens out, requests served, queues, generation rate. Pick what's actually useful for someone owning this serving stack.
3. **KV cache.** The metric (or metrics) that tell you whether you have headroom for more concurrency or you're about to evict.

We are not naming the specific metrics on purpose. Exploring `/metrics` and picking the right ones for each category is part of the work. Aim for a dashboard a teammate could open at 3 AM on a Friday night in a bar and read the picture.

The starter dashboard at `infra/grafana/provisioning/dashboards/serving.json` gives you 2 pre-built panels to build on - feel free to extend it in the Grafana UI or edit the JSON directly. The [vLLM metrics docs](https://docs.vllm.ai/en/latest/usage/metrics.html) describe what each metric means, and the [Grafana docs](https://grafana.com/docs/grafana/latest/) are your reference for building panels.

### What you should have in the end:
- Grafana dashboard covering latency, throughput, KV cache
- Every panel visibly reacts when you fire requests
- A screenshot of the full dashboard with panels reacting to a burst of requests (`screenshots/grafana_serving.png`)
- Dashboard JSON committed under `infra/grafana/provisioning/dashboards/`

---

## Phase 3 (Agent)

The goal is to build a simple self-consistency inspired agent that:
- converts an English question into a SQL query,
- runs it against a sqlite DB, 
- verifies the result makes sense, and revises if it doesn't.

  

The graph (already sketched in `agent/graph.py`):

  

```
question + schema
       │
       ▼
┌─────────────────┐
│ generate_sql    │  ── vLLM call #1
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ execute         │  (provided - runs SQL, returns rows or error)
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ verify          │  ── vLLM call #2
└─────────────────┘  asks: is this answer plausible?
       │             outputs: {ok: bool, issue: str}
       │
ok=true├──► return SQL + rows
       │
ok=false├──► ┌─────────────────┐
             │ revise          │  ── vLLM call #3 (and back to execute)
             └─────────────────┘
                    │
                    ▼
             loop (max 3 total iterations)
```

  

> **Tip:** this phase is pure agent logic - you don't need the H100 running to build the graph and draft prompts. See [Developing without the H100](#developing-without-the-h100). Do final prompt tuning against the real `Qwen3-30B-A3B` endpoint, though - behavior and tokenization differ between models.

### What to do:

  

1.  **Implement the LLM-calling nodes** in `agent/graph.py`. `generate_sql_node` is filled in as a worked example - `verify`, `revise`, and the `route_after_verify` router are yours, and each docstring spells out what the node must return. The mechanics (LLM client, graph wiring, `execute`, schema rendering) are scaffolded; the prompts and the verify/revise logic are the actual exercise.

2.  **Write the prompts** in `agent/prompts.py`. Aim for it to fire on the obvious cases: SQL errored, zero rows when the question implies rows exist, returned columns clearly don't answer the question.

3.  **Wire the conditional edge** in `agent/graph.py` so verify-false routes back into a revise step (which then re-executes). Cap the loop at 3-5 iterations.

4.  **Test interactively** with 5 questions from `evals/eval_set.jsonl`. Confirm that at least one question triggers a revise.

A way to test: 

```bash

curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "db": "..."}'
```

### What you should have in the end:
- Agent server at `http://localhost:8001`
- `verify → revise` loop wired with an iteration cap
- At least one test question that triggers a revise

---

  

## Phase 4 (Agent o11y)

  
Langfuse is running locally from your docker-compose, you need to point your agent at it.  

### What to do:

  

1. Sign up for a local Langfuse account at `http://localhost:3001`

2. Create a project, grab the public and secret keys.

3. Add them to `.env`.

4. Add the Langfuse callback handler to your LangGraph invocation:

```python

from langfuse.callback import CallbackHandler

handler =  CallbackHandler()  # picks up env vars

result = graph.invoke(state,  config={"callbacks": [handler]})

```

5. Fire 10 questions through the agent.

6. Open Langfuse UI. Find a trace and inspect it. You should see a waterfall with `generate_sql`, `verify`, and (sometimes) `revise` as nested spans, each with its prompt, response, latency, and token count.

7.  Tag your traces with metadata, you'll need it during Phase 6.

### What you should have in the end:
- Langfuse capturing traces from agent runs
- One trace inspected showing the `generate_sql` / `verify` / (sometimes) `revise` waterfall (`screenshots/langfuse_trace.png`)
- Traces tagged with metadata you'll filter on in Phase 6
- A screenshot of the trace list with your metadata tags visible (`screenshots/langfuse_tags.png`)

---

## Phase 5 (Evals)

You have 30 curated questions in `evals/eval_set.jsonl`. Each one has question text, target DB, gold SQL, expected result rows.

The eval signal is execution accuracy: run the agent's final SQL and the gold SQL against the target DB, compare result sets after canonicalizing (sort rows, ignore column-name case). Match → correct, no match → incorrect. SQL has many syntactically different ways to express the same query, but if two queries produce identical row sets on the same data, they're answering the same question.

### What to do:

1. Implement `evals/run_eval.py`. It should:

- Read the eval set

- Call the agent (HTTP) on each question

- Compute execution accuracy by running both SQLs against the target DB and comparing canonicalized row sets

-  Record how many iterations the agent took, and the pass rate at each iteration (i.e., if we had stopped after iter 0, what would pass rate be? After iter 1? Iter 2?)

- Write results to `results/eval_baseline.json`

2.  Run baseline eval. Note, this will hit your vLLM endpoint with 30 questions × ~2 vLLM calls each = ~60 requests. Watch Grafana while it runs.

3.  Look at the per-iteration pass rate. If iter 0 pass rate is the same as iter 3 pass rate, your agent architecture is doing nothing. If iter 3 is meaningfully higher, the loop is earning its keep.

### What you should have in the end:
- `evals/run_eval.py` working end-to-end
- `results/eval_baseline.json` with overall + per-iteration pass rates
- A screenshot of the Grafana dashboard while the baseline eval runs (`screenshots/grafana_eval_run.png`)
- A read on whether the agent loop is doing real work
  

---

## Phase 6 (SLOs)


This is where the configuration from Phase 1 meets reality. The target is the platform SLO from Phase 1:

> **P95 end-to-end agent latency under 5 seconds, 10+ RPS (1rps = 1 full agent run per second) over a 5-minute window.**

 
### What to do:


1.  Run the load test against your current configuration:

```bash

uv run python load_test/driver.py --rps n --duration 300

```  

Watch the Grafana dashboard while it runs.

2.  Either you hit the SLO or you don't. If you hit it on the first try - cool, but still take an iteration where you push past it to find what actually breaks. The point of this phase is the diagnosis skill, not just the green check.

3.  Diagnose. Look at the dashboard. Which metric moves first as load ramps? Where is the system spending its time? Form a specific hypothesis about what's holding you back. Don't guess at a fix until the hypothesis is grounded in something you can point at on the dashboard.

4.  Change one thing, re-run and confirm in the dashboard that the metric you targeted actually moved. Then ask whether end-to-end latency moved with it as sometimes a metric improves and the SLO doesn't, which is its own lesson.

5.  Iterate. Each iteration should produce: a one-line note in `REPORT.md` of the form *"saw X → hypothesized Y → changed Z → result was W"*, and a Grafana screenshot. Three or four iterations is normal. If you're on iteration seven, you're probably guessing instead of reading metrics - stop and re-read the dashboard.

6.  Run the eval set against your final configuration. Save to `results/eval_after_tuning.json`. If your tuning regressed quality, that's part of the writeup, analyze it.

7.  Document the full cycle in `REPORT.md`: baseline numbers, the iterations you went through, the final numbers, whether quality survived.

### What you should have in the end:
- An iteration log in `REPORT.md` of the form *"saw X → hypothesized Y → changed Z → result was W"*
- A before/after Grafana pair around the change that moved the needle (`screenshots/grafana_before.png`, `screenshots/grafana_after.png`)
- `results/eval_after_tuning.json` showing whether quality survived
- An honest verdict - SLO hit, or SLO missed with the gap quantified

---

## Phase 7 (Docs)

Wrap up `REPORT.md`. It should have:

1.  Serving configuration (Phase 1), your chosen flags, one line of justification each.

2.  Baseline eval results (Phase 5), overall pass rate, per-iteration pass rate, brief commentary.

3.  Hitting the SLO (Phase 6), baseline performance vs. SLO, the iteration log, the final numbers.

4.  Agent value, one paragraph. Did the loop actually help? How do you know? Cite the per-iteration pass rate.

5.  What you'd do with more time, and be specific here! "Add Kubernetes" doesn't count.

Aim at 2-3 pages max.

### What you should have in the end:
- `REPORT.md` complete, 2-3 pages
- All artifacts from the Final deliverables table present in the repo

## Final deliverables

By the end, your repo should contain:

| File | What it is |
|---|---|
| `REPORT.md` | Your writeup (≤ 3 pages) |
| `infra/grafana/provisioning/dashboards/serving.json` | Your Grafana dashboard with all required panels |
| `agent/graph.py`, `agent/prompts.py` | Your implemented agent |
| `evals/run_eval.py` | Your eval runner |
| `results/eval_baseline.json` | Baseline eval results |
| `results/eval_after_tuning.json` | Post-tuning eval results |
| `screenshots/vllm_manual_query.png` | vLLM serving + a manual query returning SQL (Phase 1) |
| `screenshots/grafana_serving.png` | The full Grafana dashboard with panels reacting to load (Phase 2) |
| `screenshots/langfuse_trace.png` | A Langfuse trace showing a verify→revise loop (Phase 4) |
| `screenshots/langfuse_tags.png` | The Langfuse trace list with your metadata tags visible (Phase 4) |
| `screenshots/grafana_eval_run.png` | The Grafana dashboard while the baseline eval runs (Phase 5) |
| `screenshots/grafana_before.png`, `screenshots/grafana_after.png` | Before/after the tuning change that moved the needle (Phase 6) |

---

## Grading

We want to see your thoughts and reasoning process, not the green checkmarks. Showing where you got stuck is better than omitting mentions of these points. A missed SLO with a metric-grounded diagnosis is better than a hit SLO you can't explain.

| Area | Weight | What a strong submission shows |
|---|---|---|
| **Serving config & justification** (Phase 1) | 15% | vLLM serving Qwen3-30B-A3B on the H100, with flags chosen *for this workload* (not defaults) and a one-line rationale each that shows you understood the MoE / prompt-shape / latency tradeoffs. |
| **Observability dashboard** (Phase 2) | 15% | Latency (percentiles), throughput, and KV-cache panels built from the right `/metrics`, that visibly react under load and actually answer "is it slow, and where in the request lifecycle?" Readable cold. |
| **Agent design** (Phase 3) | 10% | `verify → revise` loop wired with an iteration cap, prompts that catch the obvious failure cases, and at least one question that genuinely triggers a revise. |
| **Agent tracing** (Phase 4) | 5% | Langfuse capturing the `generate_sql / verify / (revise)` waterfall, with metadata tags you actually use in Phase 6. |
| **Eval rigor** (Phase 5) | 15% | Correct execution-accuracy comparison (canonicalized row sets), overall + per-iteration pass rate, and an honest read on whether the loop earns its keep. |
| **SLO diagnosis & iteration** (Phase 6) | 25% | A metric-grounded iteration log - *"saw X → hypothesized Y → changed Z → result W"* - with before/after evidence the targeted metric moved, and whether end-to-end latency *and* quality followed. Diagnosis quality counts more than hitting the number. |
| **Report & communication** (Phase 7) | 15% | `REPORT.md` clear, honest about misses, ≤3 pages, and "what I'd do with more time" is specific (not "add Kubernetes"). |