"""Phase 1 serving-config probe.

Drives vLLM directly with the *shape* of the Phase 3 agent workload so serving
flags can be tuned before the agent exists: each logical "run" issues 2-3
dependent chat completions (generate_sql -> verify -> sometimes revise) built
from real BIRD schemas and questions.

That matters because the Phase 1/6 SLO is stated on end-to-end agent latency,
not per-call latency. Measuring single calls would flatter the config by
hiding the fact that the SLO budget is split across dependent round trips.

Prompts put the static instructions and the DB schema *before* the question so
consecutive requests against the same DB share a cache-able prefix - the same
ordering the Phase 3 prompts must use for prefix caching to pay off.

Run:
    uv run python load_test/vllm_probe.py --rps 10 --duration 60
    uv run python load_test/vllm_probe.py --concurrency 32 --requests 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import aiohttp

from agent.schema import render_schema

ROOT = Path(__file__).resolve().parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"

SQL_SYSTEM = (
    "You are a senior data analyst. Translate the user's question into a single "
    "SQLite SELECT query. Reply with only the SQL in a ```sql code block."
)
VERIFY_SYSTEM = (
    "You check whether a SQL result plausibly answers a question. "
    'Reply with only JSON: {"ok": true|false, "issue": "..."}.'
)
# Stand-in for the executed rows the verify node will see. Kept fixed so the
# probe measures serving behaviour rather than DB variance.
FAKE_ROWS = "OK: 3 rows.\nCOLUMNS: name, total\nFIRST ROWS:\nalpha | 12\nbeta | 9\ngamma | 4"


def build_messages(kind: str, schema: str, question: str, sql: str = "") -> list[dict]:
    """Schema first, question last -> maximal shared prefix per DB."""
    if kind == "generate":
        return [
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": f"{schema}\n\n-- Question: {question}\n"},
        ]
    if kind == "verify":
        return [
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": f"{schema}\n\n-- Question: {question}\n\nSQL:\n{sql}\n\nRESULT:\n{FAKE_ROWS}\n",
            },
        ]
    return [
        {"role": "system", "content": SQL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{schema}\n\n-- Question: {question}\n\nPrevious SQL:\n{sql}\n\n"
                f"RESULT:\n{FAKE_ROWS}\n\nIssue: returned columns do not answer the question. "
                "Write a corrected query."
            ),
        },
    ]


async def one_call(
    session: aiohttp.ClientSession, url: str, model: str, messages: list[dict], max_tokens: int
) -> dict:
    """One streaming chat completion; returns TTFT and token accounting."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = None
    text_parts: list[str] = []
    usage = {}
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    text_parts.append(piece)
    total = time.monotonic() - t0
    out_tokens = usage.get("completion_tokens", 0)
    return {
        "ttft": ttft if ttft is not None else total,
        "total": total,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": out_tokens,
        # mean inter-token latency across the decode phase
        "itl": ((total - (ttft or total)) / max(out_tokens - 1, 1)) if out_tokens > 1 else 0.0,
        "text": "".join(text_parts),
    }


async def one_run(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    q: dict,
    revise: bool,
    out: list[dict],
) -> None:
    """A full synthetic agent run: generate -> verify -> (optional) revise."""
    schema = render_schema(q["db_id"])
    t0 = time.monotonic()
    calls: list[dict] = []
    try:
        gen = await one_call(
            session, args.url, args.model,
            build_messages("generate", schema, q["question"]), args.max_tokens_sql,
        )
        calls.append(gen)
        sql = gen["text"][-600:]
        ver = await one_call(
            session, args.url, args.model,
            build_messages("verify", schema, q["question"], sql), args.max_tokens_verify,
        )
        calls.append(ver)
        if revise:
            rev = await one_call(
                session, args.url, args.model,
                build_messages("revise", schema, q["question"], sql), args.max_tokens_sql,
            )
            calls.append(rev)
        status = "ok"
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"
    out.append({
        "status": status,
        "e2e": time.monotonic() - t0,
        "n_calls": len(calls),
        "ttfts": [c["ttft"] for c in calls],
        "itls": [c["itl"] for c in calls],
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
    })


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(int(round(p * (len(s) - 1))), len(s) - 1)]


async def main_async(args: argparse.Namespace) -> None:
    questions = [json.loads(l) for l in PERF_POOL.read_text().splitlines() if l.strip()]
    rnd = random.Random(args.seed)
    out: list[dict] = []

    # Warm the prefix cache so we measure steady state, not cold-start prefill.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=args.timeout),
    ) as session:
        if args.warmup:
            seen: dict[str, dict] = {}
            for q in questions:
                seen.setdefault(q["db_id"], q)
            await asyncio.gather(*[
                one_call(session, args.url, args.model,
                         build_messages("generate", render_schema(q["db_id"]), q["question"]), 8)
                for q in seen.values()
            ])
            print(f"warmed {len(seen)} schema prefixes")

        start = time.monotonic()
        tasks: list[asyncio.Task] = []

        if args.concurrency:  # closed loop: fixed in-flight runs
            sem = asyncio.Semaphore(args.concurrency)

            async def guarded(q: dict, revise: bool) -> None:
                async with sem:
                    await one_run(session, args, q, revise, out)

            for i in range(args.requests):
                tasks.append(asyncio.create_task(
                    guarded(rnd.choice(questions), rnd.random() < args.revise_rate)))
            await asyncio.gather(*tasks)
        else:  # open loop: fixed arrival rate, the SLO's actual shape
            interval = 1.0 / args.rps
            deadline = start + args.duration
            next_fire = start
            while time.monotonic() < deadline:
                tasks.append(asyncio.create_task(one_run(
                    session, args, rnd.choice(questions), rnd.random() < args.revise_rate, out)))
                next_fire += interval
                nap = next_fire - time.monotonic()
                if nap > 0:
                    await asyncio.sleep(nap)
            if tasks:
                await asyncio.wait(tasks, timeout=args.timeout)
        wall = time.monotonic() - start

    ok = [r for r in out if r["status"] == "ok"]
    e2e = [r["e2e"] for r in ok]
    ttfts = [t for r in ok for t in r["ttfts"]]
    itls = [t for r in ok for t in r["itls"]]
    summary = {
        "label": args.label,
        "mode": "closed" if args.concurrency else "open",
        "target_rps": None if args.concurrency else args.rps,
        "concurrency": args.concurrency,
        "runs": len(out),
        "ok": len(ok),
        "failed": len(out) - len(ok),
        "wall_s": round(wall, 1),
        "achieved_run_rps": round(len(ok) / wall, 2) if wall else 0,
        "e2e_p50": round(pct(e2e, 0.50), 3),
        "e2e_p95": round(pct(e2e, 0.95), 3),
        "e2e_p99": round(pct(e2e, 0.99), 3),
        "e2e_max": round(max(e2e), 3) if e2e else None,
        "ttft_p50": round(pct(ttfts, 0.50), 3),
        "ttft_p95": round(pct(ttfts, 0.95), 3),
        "itl_mean_ms": round(1000 * statistics.fmean(itls), 2) if itls else None,
        "itl_p95_ms": round(1000 * pct(itls, 0.95), 2) if itls else None,
        "out_tok_per_s": round(sum(r["completion_tokens"] for r in ok) / wall, 1) if wall else 0,
        "prompt_tok_per_s": round(sum(r["prompt_tokens"] for r in ok) / wall, 1) if wall else 0,
    }
    if out and len(ok) != len(out):
        summary["first_error"] = next(r["status"] for r in out if r["status"] != "ok")
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(args.out.read_text()) if args.out.exists() else []
        prev.append(summary)
        args.out.write_text(json.dumps(prev, indent=2))
        print(f"appended -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1") + "/chat/completions")
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    p.add_argument("--rps", type=float, default=10.0)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=0, help="closed-loop mode; overrides --rps")
    p.add_argument("--requests", type=int, default=200, help="closed-loop total runs")
    p.add_argument("--revise-rate", type=float, default=0.3, help="fraction of runs doing a 3rd call")
    p.add_argument("--max-tokens-sql", type=int, default=256)
    p.add_argument("--max-tokens-verify", type=int, default=96)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "phase1_sweep.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
