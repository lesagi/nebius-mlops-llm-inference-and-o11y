#!/usr/bin/env python3
"""Check the verify -> revise loop offline, then run it against the live agent.

    uv run python scripts/phases/phase3.py                  # asserts + 5 live questions
    uv run python scripts/phases/phase3.py --verify-only    # asserts only, no backend
    uv run python scripts/phases/phase3.py -n 10 --timeout 600

Checks: the router's truth table and the verdict parser, both offline; then
that vLLM and the agent answer, and that questions from the eval set come back
with SQL. Prints each run's node sequence so a revise is visible when it fires.

Starts `uvicorn agent.server:app` itself if :8001 is down, logging to
logs/agent.log, and stops it again on the way out.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
AGENT_LOG = ROOT / "logs" / "agent.log"

VLLM = "http://localhost:8000"
AGENT = "http://localhost:8001"
TAGS = {"phase": "3", "source": "phase3.py"}

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# scripts/ is not a package, so the repo root has to be importable before the
# offline checks can reach the graph.
sys.path.insert(0, str(ROOT))


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{OFF}   {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{OFF} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{OFF} {msg}")


# ---- Offline checks ---------------------------------------------------

def self_check() -> None:
    """Assert the two pieces of Phase 3 logic that don't need a model.

    Deliberately assert-based: this repo has no pytest and doesn't need one for
    a truth table and a regex.
    """
    from agent.graph import MAX_ITERATIONS, _extract_sql, _parse_verdict, route_after_verify
    from agent.graph import AgentState

    def state(**kw) -> AgentState:
        return AgentState(question="q", db_id="superhero", **kw)

    # The literals key add_conditional_edges in build_graph(); returning END or
    # a typo here would raise at runtime, not import time.
    assert route_after_verify(state(verify_ok=True, iteration=1)) == "end"
    assert route_after_verify(state(verify_ok=False, iteration=1)) == "revise"
    # At the cap we stop even though the verifier is still unhappy - this is the
    # only thing terminating the loop.
    assert route_after_verify(state(verify_ok=False, iteration=MAX_ITERATIONS)) == "end"
    assert route_after_verify(state(verify_ok=True, iteration=MAX_ITERATIONS)) == "end"
    ok(f"route_after_verify: 4/4 cases, cap at {MAX_ITERATIONS} iterations")

    assert _parse_verdict('{"ok": true, "issue": ""}') == (True, "")
    assert _parse_verdict('{"ok": false, "issue": "zero rows"}') == (False, "zero rows")
    assert _parse_verdict('```json\n{"ok": false, "issue": "wrong column"}\n```') == (False, "wrong column")
    assert _parse_verdict('Sure! Here is my verdict:\n{"ok": false, "issue": "errored"} Hope that helps.') == (False, "errored")
    # Fails open: an unreadable verdict is not evidence of a bug.
    assert _parse_verdict("the query looks fine to me") == (True, "")
    assert _parse_verdict('{"ok": not-json}') == (True, "")
    ok("_parse_verdict: 6/6 cases, fails open on garbage")

    assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert _extract_sql("SELECT 2") == "SELECT 2"
    # A hybrid stand-in model reasons out loud; that prose must not reach sqlite.
    assert _extract_sql("<think>hmm</think>\n```sql\nSELECT 3\n```") == "SELECT 3"
    assert _extract_sql("<think>hmm</think>SELECT 4") == "SELECT 4"
    ok("_extract_sql: 4/4 cases, drops fences and reasoning blocks")

    # An aggregate over zero matching rows is the failure mode verify kept
    # waving through, and it only reads as one if the cell says NULL.
    from agent.execution import ExecutionResult

    null_agg = ExecutionResult(ok=True, rows=[(None,)], columns=["average_crimes"], row_count=1)
    assert "NULL" in null_agg.render()
    assert "None" not in null_agg.render()
    ok("ExecutionResult.render: NULL cells render as SQL NULL")

    # Phase 6 caught a model-written query holding a threadpool thread for 512s.
    # The budget has to stop that without touching queries that behave.
    from agent.execution import QUERY_BUDGET_SECONDS, execute_sql
    from agent.schema import db_path

    if db_path("superhero").exists():
        runaway = "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r) SELECT count(*) FROM r"
        started = time.monotonic()
        aborted = execute_sql("superhero", runaway)
        elapsed = time.monotonic() - started
        assert not aborted.ok and "cancelled" in (aborted.error or ""), aborted
        assert elapsed < QUERY_BUDGET_SECONDS + 1, f"budget overshot: {elapsed:.1f}s"
        assert execute_sql("superhero", "SELECT count(*) FROM superhero").ok
        ok(f"execute_sql: a runaway query is cancelled at {QUERY_BUDGET_SECONDS:.0f}s "
           f"({elapsed:.1f}s), normal queries unaffected")
    else:
        warn("data/bird/superhero.sqlite missing - skipping the query-budget check")

    from agent.graph import REVISE_TEMPERATURE, llm

    # Generate stays greedy for reproducible evals; revise samples, or repeated
    # attempts are identical by construction.
    assert llm().temperature == 0.0
    assert llm(temperature=REVISE_TEMPERATURE).temperature > 0.0
    ok(f"llm(): generate greedy, revise samples at {REVISE_TEMPERATURE}")


# ---- Live run --------------------------------------------------------

@dataclass
class AgentProcess:
    """A uvicorn we started, so we know to stop it. None means it was already up."""

    proc: subprocess.Popen | None = None


def alive(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def start_agent(workers: int = 1) -> AgentProcess:
    """Launch the agent server if it isn't already answering.

    `workers` is Phase 6's lever: one uvicorn process serves every request from a
    single GIL, and /answer is a sync def, so a run holds a threadpool thread for
    all of its sequential model calls.
    """
    if alive(AGENT):
        ok(f"agent already serving at {AGENT}")
        return AgentProcess()

    AGENT_LOG.parent.mkdir(exist_ok=True)
    log = AGENT_LOG.open("w")
    cmd = [sys.executable, "-m", "uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "8001"]
    if workers > 1:
        cmd += ["--workers", str(workers)]
    # Own session, so stop_agent can signal the whole group. Without it a SIGKILL
    # on this parent orphans its --workers children: Phase 6 leaked four of them
    # across the 21 Aug runs, and they sat burning 5.1 of 16 cores for 19 hours.
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(60):
        if alive(AGENT, timeout=2):
            ok(f"agent started at {AGENT} (log: {AGENT_LOG.relative_to(ROOT)})")
            return AgentProcess(proc)
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    raise SystemExit(f"agent failed to start - see {AGENT_LOG}")


def stop_agent(agent: AgentProcess, timeout: int = 30) -> None:
    """Stop an agent we started, children included, without ever raising.

    A saturated agent is still draining in-flight runs when this is called - the
    Phase 6 baseline had requests taking 89s - so a bounded wait must escalate
    rather than raise. Signals go to the process *group*: uvicorn --workers means
    children, and killing only the parent is what leaked them.
    """
    if agent.proc is None:
        return
    group = os.getpgid(agent.proc.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return  # already gone, including on the SIGTERM pass
        try:
            agent.proc.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            warn(f"the agent did not stop in {timeout}s - still draining in-flight runs, killing it")
    print(f"  {DIM}stopped the agent we started, group included (flushed on shutdown){OFF}")


def answer(question: str, db_id: str, timeout: float, tags: dict[str, str]) -> dict:
    """POST one question. `tags` becomes the Langfuse trace's tags."""
    body = json.dumps({"question": question, "db": db_id, "tags": tags}).encode()
    req = urllib.request.Request(
        f"{AGENT}/answer", data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def run_questions(count: int, timeout: float) -> bool:
    questions = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()][:count]
    failures = 0
    revised = 0

    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 70}\n  {i}/{len(questions)} [{q['db_id']}] {q['question']}")
        started = time.time()
        try:
            result = answer(q["question"], q["db_id"], timeout, TAGS)
        except urllib.error.HTTPError as exc:
            bad(f"HTTP {exc.code}: {exc.read().decode()[:300]}")
            failures += 1
            continue
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            bad(f"no response after {time.time() - started:.0f}s ({exc})")
            failures += 1
            continue

        path = " -> ".join(h["node"] for h in result.get("history", []))
        if "revise" in path:
            revised += 1
        verdict = f"{GREEN}ok{OFF}" if result["ok"] else f"{RED}not ok{OFF}"
        print(f"  {verdict}  {result['iterations']} iter, {time.time() - started:.1f}s  {DIM}{path}{OFF}")
        if result.get("error"):
            print(f"  {YELLOW}error:{OFF} {result['error'][:200]}")
        print(f"  {DIM}sql:{OFF} {' '.join(result['sql'].split())[:300]}")
        rows = result.get("rows")
        if rows is not None:
            print(f"  {DIM}rows:{OFF} {len(rows)}  {DIM}first:{OFF} {str(rows[0])[:160] if rows else '-'}")

    print(f"\n{'=' * 70}")
    if failures:
        bad(f"{failures}/{len(questions)} questions did not return an answer")
    else:
        ok(f"{len(questions)}/{len(questions)} questions answered")
    if revised:
        ok(f"{revised} run(s) went through revise - the loop fires")
    else:
        warn("no run triggered a revise - Phase 3 wants at least one; try more "
             "questions, or check that verify ever returns ok=false")
    return failures == 0


def main() -> int:
    on_mac = platform.system() == "Darwin"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--count", type=int, default=5, help="questions from the eval set (default: 5)")
    # A CPU stand-in model generates at a few tokens/s, and one run is up to
    # five sequential calls, so the Mac default has to be generous.
    parser.add_argument("--timeout", type=float, default=600 if on_mac else 120,
                        help="seconds per question (default: 600 on a Mac CPU box, 120 on the H100)")
    parser.add_argument("--verify-only", action="store_true", help="offline asserts only, no backend needed")
    args = parser.parse_args()

    print("Phase 3 - verify -> revise agent")
    print("\nOffline")
    self_check()
    if args.verify_only:
        print(f"\n{GREEN}Graph logic verified.{OFF}")
        return 0

    print("\nServices")
    if not alive(VLLM):
        bad(f"vLLM unreachable at {VLLM} - run: ./scripts/start_vllm.sh")
        return 1
    ok(f"vLLM reachable at {VLLM}")
    agent = start_agent()

    try:
        passed = run_questions(args.count, args.timeout)
    finally:
        stop_agent(agent)

    print()
    if passed:
        print(f"{GREEN}Agent answering on :8001 with the loop wired.{OFF}")
    else:
        print(f"{RED}Some questions failed.{OFF} See above, and {AGENT_LOG.relative_to(ROOT)}.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
