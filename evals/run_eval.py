"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Every generate/revise attempt is scored, not just the last one, so the
per-iteration pass rate answers the Phase 5 question: had we stopped after
iteration 0, what would we have served?

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
    uv run python evals/run_eval.py --limit 3          # smoke test
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implemented in Phase 5 -------------------------------------

def matches_set(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    """Set-wise variant of matches(), reported alongside it rather than instead.

    matches() compares multisets, so a semantically correct query that omits
    DISTINCT fails it - eval question 1 returns the same coordinates once per
    race where gold returns them once. BIRD's official eval compares sets, so
    the same answer scores 1 upstream. The gap between the two rates is exactly
    the count of DISTINCT-class disagreements, which is worth measuring instead
    of arguing about; set-wise is looser rather than more correct, since gold
    question 30 legitimately returns duplicate rows that this would let a
    prediction collapse.
    """
    if gold_rows is None or pred_rows is None:
        return False
    return set(canonicalize(gold_rows)) == set(canonicalize(pred_rows))


def score_attempts(history: list[dict], db_id: str, gold_rows: list[tuple] | None) -> list[dict]:
    """Score every SQL the agent produced, paired with the verdict it drew.

    history is [{node: generate_sql, sql}, {node: verify, ok, issue}, ...], one
    sql-bearing entry per iteration, so the next verify entry after a candidate
    is that candidate's verdict. Re-running the SQL here rather than trusting
    the response's rows keeps the metric independent of agent/execution.py.
    """
    scored: list[dict] = []
    for i, entry in enumerate(history):
        if "sql" not in entry:
            continue
        verdict = next((h for h in history[i + 1:] if h.get("node") == "verify"), {})
        executed_ok, rows, error = run_sql(db_id, entry["sql"])
        scored.append({
            "iteration": len(scored),
            "node": entry["node"],
            "sql": entry["sql"],
            "executed_ok": executed_ok,
            "error": error,
            "correct": matches(gold_rows, rows) if executed_ok else False,
            "correct_setwise": matches_set(gold_rows, rows) if executed_ok else False,
            "row_count": len(rows) if rows is not None else None,
            "verify_ok": verdict.get("ok"),
            "verify_issue": verdict.get("issue", ""),
        })
    return scored


def eval_one(question: dict, agent_url: str, timeout: float = 120.0,
             tags: dict[str, str] | None = None) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    gold_executed, gold_rows, gold_error = run_sql(question["db_id"], question["gold_sql"])

    payload = {"question": question["question"], "db": question["db_id"], "tags": tags or {}}
    started = time.monotonic()
    try:
        # Explicit timeout: httpx defaults to 5s and one run is up to five
        # sequential model calls, so the default would fail every question.
        resp = httpx.post(agent_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        answer, agent_error = resp.json(), None
    except (httpx.HTTPError, json.JSONDecodeError) as e:  # one dead question, not 30
        answer, agent_error = {}, f"{type(e).__name__}: {e}"
    latency = time.monotonic() - started

    attempts = score_attempts(answer.get("history", []), question["db_id"], gold_rows)
    final = attempts[-1] if attempts else None
    return {
        "question": question["question"],
        "db_id": question["db_id"],
        "gold_sql": question["gold_sql"],
        "gold_executed": gold_executed,
        "gold_error": gold_error,
        "gold_row_count": len(gold_rows) if gold_rows is not None else None,
        "iterations": answer.get("iterations", 0),
        "latency_seconds": round(latency, 3),
        "agent_error": agent_error,
        "final_sql_error": final["error"] if final else None,
        "final_correct": bool(final and final["correct"]),
        "final_correct_setwise": bool(final and final["correct_setwise"]),
        "attempts": attempts,
    }


def _at(result: dict, k: int) -> dict | None:
    """The attempt that would have been serving at iteration k.

    Carry-forward: a question that terminated at iteration j < k stopped
    emitting, so whatever it held at j is what a poll at k would have read.
    """
    attempts = result["attempts"]
    return attempts[min(k, len(attempts) - 1)] if attempts else None


def _rate(flags) -> float:
    flags = list(flags)
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    depth = max((len(r["attempts"]) for r in results), default=0)
    by_iteration = [
        _rate(bool(_at(r, k) and _at(r, k)["correct"]) for r in results)
        for k in range(depth)
    ]
    by_iteration_setwise = [
        _rate(bool(_at(r, k) and _at(r, k)["correct_setwise"]) for r in results)
        for k in range(depth)
    ]

    # The verifier's verdicts against the truth it could not see. A loop_delta of
    # zero means something different depending on which cell is full: accepted
    # wrong answers mean the loop never got a chance, rejected correct ones mean
    # it was actively harmful. Rejections of SQL that did not execute are counted
    # here too, but they cost no model call - verify_node short-circuits them.
    verdicts = [(a["verify_ok"], a["correct"], a["executed_ok"])
                for r in results for a in r["attempts"] if a["verify_ok"] is not None]
    latencies = sorted(r["latency_seconds"] for r in results)

    return {
        "n": n,
        "answered": sum(1 for r in results if r["agent_error"] is None),
        "agent_errors": sum(1 for r in results if r["agent_error"]),
        "gold_errors": sum(1 for r in results if not r["gold_executed"]),
        "pass_rate": _rate(r["final_correct"] for r in results),
        "pass_rate_setwise": _rate(r["final_correct_setwise"] for r in results),
        "distinct_class_disagreements": sum(
            1 for r in results if r["final_correct_setwise"] and not r["final_correct"]
        ),
        "pass_rate_by_iteration": by_iteration,
        "pass_rate_by_iteration_setwise": by_iteration_setwise,
        "loop_delta": round(by_iteration[-1] - by_iteration[0], 4) if by_iteration else 0.0,
        "iteration_histogram": dict(sorted(Counter(len(r["attempts"]) for r in results).items())),
        "revised": sum(1 for r in results if any(a["node"] == "revise" for a in r["attempts"])),
        "sql_error_rate": _rate(bool(r["final_sql_error"]) for r in results),
        "verifier": {
            "accepted_correct": sum(1 for v, c, _ in verdicts if v and c),
            "accepted_wrong": sum(1 for v, c, _ in verdicts if v and not c),
            "rejected_correct": sum(1 for v, c, _ in verdicts if not v and c),
            "rejected_wrong": sum(1 for v, c, _ in verdicts if not v and not c),
            "rejected_without_executing": sum(1 for v, _, ex in verdicts if not v and not ex),
        },
        "latency_seconds": {
            "mean": round(sum(latencies) / n, 3) if n else 0.0,
            "p95": latencies[int(0.95 * (n - 1))] if n else 0.0,
            "max": latencies[-1] if n else 0.0,
        },
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per question")
    parser.add_argument("--limit", type=int, help="score only the first N questions")
    parser.add_argument("--label", help="Langfuse tag for this run (default: the --out stem)")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    questions = questions[:args.limit] if args.limit else questions
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")
    tags = {"source": "run_eval", "phase": "5", "label": args.label or args.out.stem}

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url, args.timeout, tags))
        last = results[-1]
        mark = "PASS" if last["final_correct"] else "fail"
        note = last["agent_error"] or last["final_sql_error"] or ""
        print(f"      {mark} {last['iterations']} iter {last['latency_seconds']:.1f}s {note[:80]}", flush=True)
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
