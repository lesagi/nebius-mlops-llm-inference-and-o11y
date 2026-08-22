#!/usr/bin/env python3
"""Fire tagged questions through the agent, then prove Langfuse captured them.

    uv run python scripts/phases/phase4.py                  # 10 tagged runs, then verify
    uv run python scripts/phases/phase4.py --verify-only    # check the stack + stored traces
    uv run python scripts/phases/phase4.py -n 3

Checks: Langfuse answers and the keys authenticate, the traces come back tagged,
no trace arrived truncated, and at least one carries the full
generate_sql / verify / revise waterfall. Then prints the two URLs to screenshot.

Reuses phase3.py's agent process management. Backend-agnostic: the trace shape
does not depend on model quality, so the CPU stand-in is fine here - but 10
questions against it takes 10-15 minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# The process management, health polling and /answer client are all phase3's,
# and duplicating them here would be two copies to keep in step.
from phase3 import (  # noqa: E402
    DIM, GREEN, OFF, RED, YELLOW, AgentProcess, alive, answer, bad, ok, start_agent, warn,
)

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
LANGFUSE = os.environ.get("LANGFUSE_HOST", "http://localhost:3001")
VLLM = "http://localhost:8000"
TAGS = {"phase": "4", "source": "phase4.py"}
# server.py flattens the tag dict to "key:value" for Langfuse's tag column.
TAG_FILTER = [f"{k}:{v}" for k, v in sorted(TAGS.items())]

# The nodes the Phase 4 screenshot has to show. revise is separate because it
# only appears when the loop actually fires.
CORE_SPANS = {"generate_sql", "verify"}

# Langfuse's ingestion worker lags the API accepting the batch.
ATTEMPTS, DELAY = 6, 5.0


def check_langfuse() -> bool:
    print("\nLangfuse")
    good = True
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        if os.environ.get(key):
            ok(f"{key} set")
        else:
            bad(f"{key} missing from .env - sign up at {LANGFUSE} and create a project")
            good = False
    try:
        with urllib.request.urlopen(f"{LANGFUSE}/api/public/health", timeout=5) as resp:
            health = json.load(resp)
        ok(f"{LANGFUSE} healthy (v{health.get('version', '?')})")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        bad(f"{LANGFUSE} unreachable ({exc}) - run: docker compose up -d")
        good = False
    return good


def client():
    from langfuse import get_client

    return get_client()


def span_names(lf, trace) -> list[str]:
    """Observation names for a trace.

    trace.list returns observation *ids*; only trace.get inflates them into
    objects with names.
    """
    obs = trace.observations
    if obs and isinstance(obs[0], str):
        obs = lf.api.trace.get(trace.id).observations
    return [o.name for o in obs]


def fire(count: int, timeout: float) -> int:
    questions = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()][:count]
    print(f"\nFiring {len(questions)} questions, tagged {TAG_FILTER}")
    fired = 0
    for i, q in enumerate(questions, 1):
        started = time.time()
        try:
            result = answer(q["question"], q["db_id"], timeout, TAGS)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError) as exc:
            bad(f"{i}/{len(questions)} [{q['db_id']}] failed: {exc}")
            continue
        fired += 1
        path = " -> ".join(h["node"] for h in result.get("history", []))
        print(f"  {i}/{len(questions)} [{q['db_id']}] {result['iterations']} iter, "
              f"{time.time() - started:.1f}s  {DIM}{path}{OFF}")
    return fired


def verify_traces(expected: int) -> bool:
    print("\nTraces")
    lf = client()
    # The batch processor ships every 5s, so read-after-write needs this.
    lf.flush()
    traces = lf.api.trace.list(tags=TAG_FILTER, limit=50, order_by="timestamp.desc").data
    if not traces:
        if not expected:
            # --verify-only against a Langfuse that has never seen a run. Not a
            # wiring problem, just nothing to inspect yet.
            bad(f"no traces tagged {TAG_FILTER} yet - run without --verify-only to fire some")
        else:
            bad(f"fired {expected} but nothing came back tagged {TAG_FILTER} - "
                "is the agent passing tags through?")
        return False
    ok(f"{len(traces)} trace(s) tagged {TAG_FILTER}")
    if expected and len(traces) < expected:
        warn(f"fired {expected} but only {len(traces)} came back - some are still in flight")

    # Only the runs we just fired can regress. With nothing fired, look at the
    # newest few so --verify-only stays fast and doesn't poll for old traces.
    recent = traces[:expected or 10]

    # Langfuse ingests through a worker queue, so flush() only means the batch
    # reached the server - the name and the observations become queryable a
    # little later. Poll rather than assert, or a healthy run fails on a race.
    # A name that never arrives is the real bug this guards: it means the root
    # span was dropped, which is what an unflushed shutdown looks like.
    for attempt in range(ATTEMPTS):
        spans = {t.id: set(span_names(lf, t)) for t in recent}
        unready = {
            t.id: ("no root span" if not t.name else f"missing {sorted(CORE_SPANS - spans[t.id])}")
            for t in recent
            if not t.name or not CORE_SPANS <= spans[t.id]
        }
        if not unready:
            break
        if attempt + 1 == ATTEMPTS:
            for tid, why in unready.items():
                bad(f"trace {tid}: {why} after {ATTEMPTS * DELAY:.0f}s "
                    f"(has {sorted(spans[tid])})")
            return False
        print(f"  {DIM}{len(unready)} trace(s) still materializing, "
              f"retrying in {DELAY:.0f}s{OFF}")
        time.sleep(DELAY)
        recent = lf.api.trace.list(
            tags=TAG_FILTER, limit=50, order_by="timestamp.desc",
        ).data[:len(recent)]

    ok(f"every recent trace has a root span, named {recent[0].name!r}")
    ok(f"every recent trace has {sorted(CORE_SPANS)} spans")
    best = next((t for t in recent if "revise" in spans[t.id]), None)

    if best is None:
        warn("no trace shows a revise - the trace screenshot wants one; run more questions")
        best = recent[0]
    else:
        ok(f"trace {best.id} shows the full generate_sql / verify / revise waterfall")

    print(f"\n{'=' * 70}")
    print("  Screenshot 1 - screenshots/langfuse_trace.png")
    print(f"  {LANGFUSE}{best.html_path}")
    print(f"  {DIM}expand the waterfall so the nested ChatOpenAI generations,{OFF}")
    print(f"  {DIM}their latencies and token counts are visible{OFF}")
    print("\n  Screenshot 2 - screenshots/langfuse_tags.png")
    print(f"  {LANGFUSE}/project/{best.html_path.split('/')[2]}/traces")
    print(f"  {DIM}filter on a tag and keep the Tags column in frame{OFF}")
    print(f"{'=' * 70}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--count", type=int, default=10, help="questions to fire (default: 10)")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per question (default: 600)")
    parser.add_argument("--verify-only", action="store_true", help="skip firing, check the stored traces")
    args = parser.parse_args()

    print("Phase 4 - Langfuse tracing")
    if not check_langfuse():
        return 1

    if args.verify_only:
        return 0 if verify_traces(expected=0) else 1

    print("\nServices")
    if not alive(VLLM):
        bad(f"vLLM unreachable at {VLLM} - run: ./scripts/start_vllm.sh")
        return 1
    ok(f"vLLM reachable at {VLLM}")
    agent = start_agent()

    try:
        fired = fire(args.count, args.timeout)
    finally:
        if agent.proc is not None:
            # Graceful stop, so the lifespan hook gets to flush. A kill here is
            # exactly what truncated the last trace of every Phase 3 run.
            agent.proc.terminate()
            agent.proc.wait(timeout=30)
            print(f"  {DIM}stopped the agent we started (flushed on shutdown){OFF}")

    if not fired:
        bad("nothing was fired successfully")
        return 1
    passed = verify_traces(expected=fired)
    print()
    if passed:
        print(f"{GREEN}Langfuse is capturing tagged traces. Take the two screenshots above.{OFF}")
    else:
        print(f"{RED}Tracing is not complete.{OFF} See failures above.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
