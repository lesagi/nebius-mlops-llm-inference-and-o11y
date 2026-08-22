#!/usr/bin/env python3
"""Verify the Grafana serving dashboard, then drive load through it.

    uv run python scripts/phases/phase2.py                  # verify + 5 min of load
    uv run python scripts/phases/phase2.py --verify-only    # just the checks, no load
    uv run python scripts/phases/phase2.py --rps 10 --duration 300

Checks: every PromQL expression in serving.json resolves against the live
Prometheus, the datasource scrape interval matches Prometheus, and after the
load run every panel that should have moved actually did.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "infra" / "grafana" / "provisioning" / "dashboards" / "serving.json"
PROM_CONFIG = ROOT / "infra" / "prometheus.yml"
DATASOURCE = ROOT / "infra" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"

PROM = "http://localhost:9090"
GRAFANA = "http://localhost:3000"
DASHBOARD_URL = f"{GRAFANA}/d/vllm-serving"

# $__rate_interval is a Grafana macro; Prometheus needs a literal window.
RATE_INTERVAL = "1m"
# Panels a load run must move. The rest (preemptions, KV) are legitimately flat
# on a healthy box, so demanding data from them would fail a passing system.
MUST_HAVE_DATA = {
    "End-to-end request latency",
    "Where the time goes (mean, stacked to e2e)",
    "Time to first token",
    "Inter-token latency",
    "Token throughput",
    "Requests finished / s, by outcome",
    "Batch fullness (tokens per engine step)",
    "Tokens per request",
    "Prefix cache hit rate",
}

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{OFF}   {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{OFF} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{OFF} {msg}")


def env_from_dotenv() -> dict[str, str]:
    """Read .env. The probe itself only reads os.environ, so without this it
    silently falls back to the H100 checkpoint id and vLLM answers an unserved
    model with a bare 404 that looks like a bad URL."""
    found: dict[str, str] = {}
    for name in (".env",):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            found[key.strip()] = value.strip().strip('"').strip("'")
    return found


def get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def promql(expr: str) -> dict:
    return get(f"{PROM}/api/v1/query?{urllib.parse.urlencode({'query': expr})}")


def panel_targets(dashboard: dict) -> list[tuple[dict, dict]]:
    """Every PromQL expression in the dashboard, panels and annotations alike.

    Annotations are easy to forget: a broken annotation query renders as no
    annotation at all, which is indistinguishable from a healthy system."""
    found = [(p, t) for p in dashboard["panels"] for t in p.get("targets", [])]
    for anno in dashboard.get("annotations", {}).get("list", []):
        target = anno.get("target", {})
        if target.get("expr"):
            found.append(({"title": f"annotation: {anno.get('name', '?')}"}, target))
    return found


def check_services() -> bool:
    print("\nServices")
    good = True
    for name, url in (("Prometheus", f"{PROM}/api/v1/status/config"), ("Grafana", f"{GRAFANA}/api/health")):
        try:
            get(url, timeout=5)
            ok(f"{name} reachable")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            bad(f"{name} unreachable at {url} ({exc}) - run: docker compose up -d")
            good = False

    # `docker compose restart prometheus` returns before Prometheus has loaded its
    # config and run a first scrape, so checking immediately after sees no targets
    # at all, or one with health "unknown". Wait it out rather than failing a
    # healthy stack for being three seconds young.
    #
    # "down" needs the same patience: restart_vllm.sh returns as soon as /health
    # answers, but the target keeps the verdict of its last scrape - taken while
    # vLLM was still starting - until the next one lands up to a scrape_interval
    # later. Breaking on anything-but-unknown failed a healthy server for being
    # ten seconds young, which is exactly what happened to the api-server-count 8
    # run.
    vllm, deadline = [], time.time() + 45
    while time.time() < deadline:
        try:
            targets = get(f"{PROM}/api/v1/targets")["data"]["activeTargets"]
            vllm = [t for t in targets if t["labels"].get("job") == "vllm"]
        except (urllib.error.URLError, OSError, KeyError, TimeoutError):
            vllm = []
        if vllm and vllm[0]["health"] == "up":
            break
        if sys.stdout.isatty():
            state = vllm[0]["health"] if vllm else "no target"
            print(f"  {DIM}...waiting for Prometheus to scrape the vllm target ({state}){OFF}", end="\r")
        time.sleep(3)
    if sys.stdout.isatty():
        print(" " * 60, end="\r")

    if not vllm:
        bad("Prometheus has no 'vllm' scrape target after 45s - check infra/prometheus.yml")
        good = False
    elif vllm[0]["health"] != "up":
        bad(f"vLLM scrape target is {vllm[0]['health']}: {vllm[0].get('lastError')}")
        bad("  vLLM not running, or /metrics is timing out - see the scrape success line below")
        good = False
    else:
        ok(f"vLLM scrape target up ({vllm[0]['scrapeUrl']})")

    # A timing-out scrape is the worst failure this stack has: the dashboard goes
    # blank exactly when load is highest, and empty panels read as "no traffic".
    try:
        res = promql('avg_over_time(up{job="vllm"}[15m]) * 100')["data"]["result"]
        if res:
            rate = float(res[0]["value"][1])
            msg = f"scrape success over the last 15m: {rate:.1f}%"
            if rate < 95:
                warn(f"{msg} - gaps on the dashboard are MISSING DATA, not idle time")
                warn("  raise scrape_timeout in infra/prometheus.yml, or reduce load on the API server")
                warn("  (backward-looking window: stays low for 15m after a fix lands)")
            else:
                ok(msg)
    except (urllib.error.URLError, OSError, KeyError, ValueError, TimeoutError):
        warn("could not read scrape success rate")
    return good


def check_scrape_alignment() -> None:
    """$__rate_interval is derived from the datasource's timeInterval, not from
    Prometheus. If they disagree every rate window on the dashboard is wrong."""
    scrape = re.search(r"scrape_interval:\s*(\S+)", PROM_CONFIG.read_text())
    declared = re.search(r"timeInterval:\s*['\"]?(\S+?)['\"]?\s*$", DATASOURCE.read_text(), re.M)
    if not scrape or not declared:
        warn("could not read scrape_interval / timeInterval - skipping alignment check")
    elif scrape.group(1) != declared.group(1):
        bad(f"datasource timeInterval={declared.group(1)} but Prometheus scrape_interval={scrape.group(1)}")
    else:
        ok(f"datasource timeInterval matches scrape_interval ({scrape.group(1)})")


def check_expressions(dashboard: dict, require_data: bool, window: str = RATE_INTERVAL) -> bool:
    label = "expressions return data" if require_data else "expressions are valid PromQL"
    print(f"\nDashboard: {label}")
    failures, empty = [], []

    for panel, target in panel_targets(dashboard):
        expr = target["expr"].replace("$__rate_interval", window)
        where = f"{panel['title']} [{target['refId']}]"
        try:
            result = promql(expr)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            failures.append(f"{where}: {exc}")
            continue
        if result.get("status") != "success":
            failures.append(f"{where}: {result.get('error')}")
            continue
        finite = [
            s["value"][1] for s in result["data"]["result"]
            if s["value"][1] not in ("NaN", "+Inf", "-Inf")
        ]
        if not finite:
            empty.append(where)
            if require_data and panel["title"] in MUST_HAVE_DATA:
                failures.append(f"{where}: no data while load is running")

    total = len(panel_targets(dashboard))
    for message in failures:
        bad(message)
    if not failures:
        ok(f"{total}/{total} expressions across {len(dashboard['panels'])} panels")
    if empty and not require_data:
        print(f"  {DIM}{len(empty)} idle (no traffic in window) - expected before load{OFF}")
    return not failures


def run_load(model: str, rps: float, duration: int) -> None:
    # The stat tiles average over a fixed 5-minute window, so they only read true
    # once the run has been going that long.
    ends = time.strftime("%H:%M:%S", time.localtime(time.time() + duration))

    print(f"\n{'=' * 70}")
    print(f"  Load: {rps} runs/s for {duration}s, ending {ends}")
    print(f"  {DASHBOARD_URL}?refresh=5s&from=now-5m&to=now")
    print(f"{'=' * 70}\n")

    cmd = [
        sys.executable, str(ROOT / "load_test" / "vllm_probe.py"),
        "--model", model,
        "--rps", str(rps),
        "--duration", str(duration),
        "--out", str(ROOT / "results" / "phase2_load.json"),
        "--label", "phase2-dashboard",
    ]
    subprocess.run(cmd, cwd=ROOT, check=False)


def main() -> int:
    on_mac = platform.system() == "Darwin"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rps", type=float, default=0.5 if on_mac else 10.0,
                        help="agent runs per second (default: 0.5 on a Mac CPU box, 10 on the H100)")
    parser.add_argument("--duration", type=int, default=300, help="seconds of load (default: 300)")
    parser.add_argument("--verify-only", action="store_true", help="run the checks, skip the load")
    args = parser.parse_args()

    dashboard = json.loads(DASHBOARD.read_text())
    print(f"Phase 2 - {dashboard['title']} ({DASHBOARD.relative_to(ROOT)})")

    # MUST_HAVE_DATA matches on panel title, so renaming a panel would silently
    # drop it from the post-load check instead of failing. Catch that here.
    titles = {p["title"] for p in dashboard["panels"]}
    stale = MUST_HAVE_DATA - titles
    if stale:
        bad(f"MUST_HAVE_DATA names panels that no longer exist: {sorted(stale)}")
        bad("A panel was renamed - update MUST_HAVE_DATA or the check is a no-op.")
        return 1

    if not check_services():
        return 1
    check_scrape_alignment()
    if not check_expressions(dashboard, require_data=False):
        return 1

    if args.verify_only:
        print(f"\n{GREEN}Dashboard verified.{OFF}")
        return 0

    env = env_from_dotenv()
    model = os.environ.get("VLLM_MODEL") or env.get("VLLM_MODEL")
    if not model:
        bad("VLLM_MODEL not set in the environment or .env - cannot drive load")
        return 1
    print(f"\n  model: {model}")

    run_load(model, args.rps, args.duration)

    # 5m window, not 1m: the probe drains in-flight requests after the timer ends,
    # so on a slow box the last minute can be quiet even though the run succeeded.
    passed = check_expressions(dashboard, require_data=True, window="5m")
    print()
    if passed:
        print(f"{GREEN}Every panel reacted to load.{OFF}")
    else:
        print(f"{RED}Some panels stayed empty under load.{OFF} See failures above.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
