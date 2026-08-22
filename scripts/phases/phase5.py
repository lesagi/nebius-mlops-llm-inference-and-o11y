#!/usr/bin/env python3
"""Check the eval scoring offline, then run the baseline eval through the agent.

    uv run python scripts/phases/phase5.py                  # asserts + all 30 questions
    uv run python scripts/phases/phase5.py --verify-only    # asserts only, no backend
    uv run python scripts/phases/phase5.py --limit 3        # smoke test
    uv run python scripts/phases/phase5.py --out results/eval_schema_values.json \
        --label schema-values

Checks the two pieces of Phase 5 logic that need no model - the per-iteration
carry-forward and the multiset/set-wise split - then drives evals/run_eval.py
and reads the summary back out: overall pass rate, the per-iteration series, and
whether the verify -> revise loop moved it.

30 questions is ~5 minutes on the H100 and ~40 on a CPU stand-in, hence --limit.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# phase3 owns the agent process management, health polling and /answer client.
from phase3 import (  # noqa: E402
    DIM, GREEN, OFF, RED, YELLOW, alive, bad, ok, start_agent, warn,
)

RUN_EVAL = ROOT / "evals" / "run_eval.py"
DEFAULT_OUT = ROOT / "results" / "eval_baseline.json"
VLLM = "http://localhost:8000"
AGENT = "http://localhost:8001"
GRAFANA = "http://localhost:3000/d/vllm-serving"


# ---- Offline checks ---------------------------------------------------

def _synthetic(*attempts: tuple[bool, bool]) -> dict:
    """An eval_one()-shaped result from (correct, verify_ok) pairs."""
    return {
        "attempts": [
            {"iteration": i, "node": "generate_sql" if i == 0 else "revise",
             "sql": f"SELECT {i}", "executed_ok": True, "error": None,
             "correct": correct, "correct_setwise": correct, "row_count": 1,
             "verify_ok": verdict, "verify_issue": ""}
            for i, (correct, verdict) in enumerate(attempts)
        ],
        "agent_error": None, "gold_executed": True, "latency_seconds": 1.0,
        "final_correct": attempts[-1][0], "final_correct_setwise": attempts[-1][0],
        "final_sql_error": None, "iterations": len(attempts),
    }


def self_check() -> None:
    """Assert the scoring logic. No model, no agent, no sqlite."""
    from evals.run_eval import _at, matches, matches_set, score_attempts, summarize

    # One question stops at iteration 0 with a correct answer; the other needs
    # all three and only gets there at the end. The first must keep its answer at
    # every later iteration, or the series reads as a regression it never had.
    stopped_early = _synthetic((True, True))
    went_the_distance = _synthetic((False, False), (False, False), (True, True))
    summary = summarize([stopped_early, went_the_distance])

    assert summary["pass_rate_by_iteration"] == [0.5, 0.5, 1.0], summary["pass_rate_by_iteration"]
    assert summary["loop_delta"] == 0.5
    assert summary["pass_rate"] == 1.0
    assert summary["iteration_histogram"] == {1: 1, 3: 1}
    assert summary["revised"] == 1
    ok("summarize: carry-forward holds a terminated answer at later iterations")

    assert summary["verifier"] == {
        "accepted_correct": 2, "accepted_wrong": 0,
        "rejected_correct": 0, "rejected_wrong": 2,
        "rejected_without_executing": 0,
    }, summary["verifier"]
    ok("summarize: verifier 2x2 counts verdicts against measured truth")

    # A question the agent never answered has no attempts, and must count as
    # wrong at every iteration rather than crashing the aggregate.
    assert _at({"attempts": []}, 0) is None
    assert summarize([{**_synthetic((True, True)), "attempts": []}])["pass_rate_by_iteration"] == []
    ok("summarize: an unanswered question scores zero, it does not raise")

    # Eval question 1: gold DISTINCTs, the model returns the same row per race.
    assert matches([(1,)], [(1,), (1,)]) is False
    assert matches_set([(1,)], [(1,), (1,)]) is True
    # The other direction, which is why set-wise is reported and not substituted:
    # gold question 30 legitimately returns duplicates.
    assert matches_set([(1,), (1,)], [(1,)]) is True
    ok("matches vs matches_set: the DISTINCT disagreement is measured both ways")

    history = [
        {"node": "generate_sql", "sql": "SELECT 1"},
        {"node": "verify", "ok": False, "issue": "zero rows"},
        {"node": "revise", "sql": "SELECT 2", "issue": "zero rows"},
        {"node": "verify", "ok": True, "issue": ""},
    ]
    attempts = score_attempts(history, "superhero", None)
    assert [a["iteration"] for a in attempts] == [0, 1]
    assert [a["node"] for a in attempts] == ["generate_sql", "revise"]
    # Each candidate carries the verdict that followed it, not the one before.
    assert [a["verify_ok"] for a in attempts] == [False, True]
    assert attempts[0]["verify_issue"] == "zero rows"
    ok("score_attempts: one entry per iteration, paired with the verdict it drew")

    # The other half of Phase 5: the literals that were missing from the prompt.
    from agent import schema as schema_mod

    if not schema_mod.db_path("toxicology").exists():
        warn("data/bird/toxicology.sqlite missing - skipping the schema-values check")
        return
    was = schema_mod.SAMPLE_VALUES
    try:
        schema_mod.SAMPLE_VALUES = True
        schema_mod.render_schema.cache_clear()
        with_values = schema_mod.render_schema("toxicology")
        schema_mod.SAMPLE_VALUES = False
        schema_mod.render_schema.cache_clear()
        without_values = schema_mod.render_schema("toxicology")
    finally:
        schema_mod.SAMPLE_VALUES = was
        schema_mod.render_schema.cache_clear()

    # The exact two literals the 30B guessed wrong in Phase 4.
    assert "'cl'" in with_values and "'+'" in with_values
    # The comma separates columns, so it has to land before the comment or the
    # DDL reads as malformed to the model.
    assert '"element" TEXT,  -- values:' in with_values
    assert "-- values:" not in without_values
    ok("render_schema: value comments carry the missing literals, and switch off cleanly")


# ---- Reading the result ----------------------------------------------

def _rel(path: Path) -> str:
    """--out is allowed to point outside the repo, where relative_to() raises."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def report(out_file: Path) -> bool:
    payload = json.loads(out_file.read_text())
    s = payload["summary"]
    n = s["n"]

    print(f"\n{'=' * 70}")
    print(f"  {_rel(out_file)}  -  {n} questions, {payload['wall_clock_seconds']:.0f}s")
    print(f"{'=' * 70}")
    print(f"  pass rate (multiset)  {s['pass_rate']:.3f}  ({round(s['pass_rate'] * n)}/{n})")
    print(f"  pass rate (set-wise)  {s['pass_rate_setwise']:.3f}  "
          f"({round(s['pass_rate_setwise'] * n)}/{n})  "
          f"{DIM}{s['distinct_class_disagreements']} DISTINCT-class disagreement(s){OFF}")
    series = "  ".join(f"iter{i} {r:.3f}" for i, r in enumerate(s["pass_rate_by_iteration"]))
    print(f"  per iteration         {series}")
    setwise = s["pass_rate_by_iteration_setwise"]
    if setwise != s["pass_rate_by_iteration"]:
        print("  " + " " * 20 + "  set-wise: "
              + "  ".join(f"iter{i} {r:.3f}" for i, r in enumerate(setwise)))
    v = s["verifier"]
    print(f"  verifier              accepted {v['accepted_correct']} correct / "
          f"{v['accepted_wrong']} wrong,  rejected {v['rejected_correct']} correct / "
          f"{v['rejected_wrong']} wrong")
    lat = s["latency_seconds"]
    print(f"  latency               mean {lat['mean']:.1f}s  p95 {lat['p95']:.1f}s  max {lat['max']:.1f}s")
    print(f"  iterations            {s['iteration_histogram']}  revised: {s['revised']}")
    print(f"{'=' * 70}\n")

    if s["agent_errors"]:
        bad(f"{s['agent_errors']}/{n} questions got no answer from the agent")
    if s["gold_errors"]:
        bad(f"{s['gold_errors']} gold queries failed to execute - the metric is broken, not the model")
    if s["sql_error_rate"]:
        warn(f"{s['sql_error_rate']:.1%} of final answers did not execute at all")

    # The Phase 5 question. A zero delta means different things depending on
    # which verifier cell is full, so name the cell rather than the number.
    delta, moved = s["loop_delta"], round(s["loop_delta"] * n)
    if delta > 0:
        ok(f"the loop earns its keep: +{delta:.3f} pass rate, {moved} question(s) fixed by revising")
    elif delta < 0:
        bad(f"the loop is net harmful: {delta:.3f} pass rate, it broke {-moved} question(s)")
    elif v["accepted_wrong"]:
        warn(f"the loop is decoration: verify accepted {v['accepted_wrong']} wrong answer(s), "
             "so revise never fired on them - the verifier is the constraint, not the iteration cap")
    else:
        warn("the loop is decoration: no iteration changed a verdict")
    if v["rejected_correct"]:
        warn(f"verify rejected {v['rejected_correct']} answer(s) that were already correct")
    return s["agent_errors"] == 0 and s["gold_errors"] == 0


# ---- Main ------------------------------------------------------------

def main() -> int:
    on_mac = platform.system() == "Darwin"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT.name}")
    parser.add_argument("--label", help="Langfuse tag for the run (default: the --out stem)")
    parser.add_argument("--limit", type=int, help="score only the first N questions")
    parser.add_argument("--timeout", type=float, default=600 if on_mac else 120,
                        help="seconds per question (default: 600 on a Mac CPU box, 120 on the H100)")
    parser.add_argument("--verify-only", action="store_true", help="offline asserts only, no backend needed")
    parser.add_argument("--report-only", action="store_true", help="re-read an existing --out, fire nothing")
    args = parser.parse_args()

    print("Phase 5 - execution-accuracy eval")
    print("\nOffline")
    self_check()
    if args.verify_only:
        print(f"\n{GREEN}Scoring logic verified.{OFF}")
        return 0
    if args.report_only:
        return 0 if report(args.out) else 1

    print("\nServices")
    if not alive(VLLM):
        bad(f"vLLM unreachable at {VLLM} - run: ./scripts/start_vllm.sh")
        return 1
    ok(f"vLLM reachable at {VLLM}")
    if not alive("http://localhost:3000", timeout=3):
        warn("Grafana not answering on :3000 - the eval will run, but there is no screenshot to take")
    agent = start_agent()
    from agent.schema import SAMPLE_VALUES  # same env, so the same rule as the agent

    mode = json.loads(urllib.request.urlopen(f"{AGENT}/health", timeout=5).read()).get("schema_values")
    wanted = "on" if SAMPLE_VALUES else "off"
    if mode != wanted:
        warn(f"the agent renders schemas with values {mode}, but SCHEMA_SAMPLE_VALUES asks for "
             f"{wanted} - it was already running, so restart it or the comparison is meaningless")
    ok(f"agent renders schemas with sampled values: {mode}")

    print(f"\n{'=' * 70}")
    print(f"  Screenshot - screenshots/grafana_eval_run.png")
    print(f"  {GRAFANA}?refresh=5s&from=now-15m&to=now")
    print(f"  {DIM}capture it WHILE the eval runs; watch 'Requests finished / s, by")
    print(f"  outcome' for finished_reason=\"length\" - a truncated SQL fails for")
    print(f"  reasons that have nothing to do with the model's SQL{OFF}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, str(RUN_EVAL),
        "--out", str(args.out),
        "--agent-url", f"{AGENT}/answer",
        "--timeout", str(args.timeout),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.label:
        cmd += ["--label", args.label]

    try:
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
    finally:
        if agent.proc is not None:
            agent.proc.terminate()
            agent.proc.wait(timeout=10)
            print(f"  {DIM}stopped the agent we started{OFF}")

    if completed.returncode != 0 or not args.out.exists():
        print(f"\n{RED}run_eval.py failed.{OFF} See the output above.")
        return 1

    # A low pass rate is a finding, not a broken script, so it does not fail here.
    healthy = report(args.out)
    print(f"{GREEN}Eval complete:{OFF} {_rel(args.out)}"
          if healthy else f"{YELLOW}Eval ran, but the harness reported problems above.{OFF}")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
