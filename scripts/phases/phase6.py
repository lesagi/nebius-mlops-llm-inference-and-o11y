#!/usr/bin/env python3
"""Drive the agent at the SLO arrival rate, then diagnose what held it back.

    uv run python scripts/phases/phase6.py                        # 10 rps, 300s
    uv run python scripts/phases/phase6.py --rps 20 --label push  # find the break point
    uv run python scripts/phases/phase6.py --workers 4 --label workers-4
    uv run python scripts/phases/phase6.py --verify-only          # stack + config only
    uv run python scripts/phases/phase6.py --report-only          # re-read recorded runs

The SLO is p95 end-to-end *agent* latency under 5s at 10+ full runs/s over a
5-minute window. Everything before this phase measured either the wrong layer
(vllm_probe.py replays calls, it is not an agent run) or the wrong concurrency
(the Phase 5 eval is sequential).

The agent exports no metrics of its own and does not need to: driver.py gives the
client-side percentiles, Prometheus gives vLLM's own per-call latency, and the
gap between them is the agent's overhead - FastAPI, LangGraph, sqlite, the
Langfuse callback, and any queueing inside the uvicorn process.

Every run is appended to results/phase6_slo.json with the configuration it ran
under, because an iteration log whose rows cannot be attributed is worthless.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# phase2 owns the Prometheus client and the service checks; phase3 owns the agent
# process management. Duplicating either would be two copies to keep in step.
from phase2 import GRAFANA, check_services, promql  # noqa: E402
from phase3 import DIM, GREEN, OFF, RED, YELLOW, alive, bad, ok, start_agent, stop_agent, warn  # noqa: E402
from phase5 import _rel  # noqa: E402

DRIVER = ROOT / "load_test" / "driver.py"
VLLM_LOG = ROOT / "logs" / "vllm.log"
SLO_FILE = ROOT / "results" / "phase6_slo.json"
DASHBOARD_URL = f"{GRAFANA}/d/vllm-serving"
AGENT = "http://localhost:8001"

# The SLO, as one place to change it.
SLO_P95_SECONDS = 5.0
SLO_RPS = 10.0

# Lifted verbatim from serving.json with $__rate_interval resolved to the run
# length, so these numbers and the screenshot cannot disagree. 5m is both the
# default run duration and the window the stat tiles average over.
W = "5m"
QUERIES = {
    "vllm_call_p50": f"histogram_quantile(0.50, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[{W}])))",
    "vllm_call_p95": f"histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[{W}])))",
    "calls_per_second": f"sum(rate(vllm:request_success_total[{W}]))",
    "queue_mean": f"sum(rate(vllm:request_queue_time_seconds_sum[{W}])) / sum(rate(vllm:request_queue_time_seconds_count[{W}]))",
    "prefill_mean": f"sum(rate(vllm:request_prefill_time_seconds_sum[{W}])) / sum(rate(vllm:request_prefill_time_seconds_count[{W}]))",
    "decode_mean": f"sum(rate(vllm:request_decode_time_seconds_sum[{W}])) / sum(rate(vllm:request_decode_time_seconds_count[{W}]))",
    "unaccounted_mean": (
        f"clamp_min((sum(rate(vllm:e2e_request_latency_seconds_sum[{W}])) "
        f"- sum(rate(vllm:request_queue_time_seconds_sum[{W}])) "
        f"- sum(rate(vllm:request_prefill_time_seconds_sum[{W}])) "
        f"- sum(rate(vllm:request_decode_time_seconds_sum[{W}]))) "
        f"/ sum(rate(vllm:e2e_request_latency_seconds_count[{W}])), 0)"
    ),
    # Averaged over the window, not read as instant gauges: by the time this runs
    # the driver has drained and the engine is idle, so an instant read describes
    # the wrong moment. The workers-4 row recorded running 0.0 that way.
    "running": f"sum(avg_over_time(vllm:num_requests_running[{W}]))",
    "waiting_peak": f"sum(max_over_time(vllm:num_requests_waiting[{W}]))",
    "batch_tokens_p95": f"histogram_quantile(0.95, sum by (le) (rate(vllm:iteration_tokens_total_bucket[{W}])))",
    "output_tokens_p95": f"histogram_quantile(0.95, sum by (le) (rate(vllm:request_generation_tokens_bucket[{W}])))",
    "kv_used": f"max(max_over_time(vllm:kv_cache_usage_perc[{W}]))",
    "prefix_hit_rate": f"sum(rate(vllm:prefix_cache_hits_total[{W}])) / sum(rate(vllm:prefix_cache_queries_total[{W}]))",
    "finished_length_share": (
        f'sum(rate(vllm:request_success_total{{finished_reason="length"}}[{W}])) '
        f"/ sum(rate(vllm:request_success_total[{W}]))"
    ),
    "preemptions_per_second": f"sum(rate(vllm:num_preemptions_total[{W}]))",
    "scrape_success_percent": f'avg_over_time(up{{job="vllm"}}[{W}]) * 100',
}


def scalar(expr: str) -> float | None:
    """First finite sample of an instant query, or None."""
    try:
        result = promql(expr)["data"]["result"]
    except (urllib.error.URLError, OSError, KeyError, TimeoutError):
        return None
    for series in result:
        value = series["value"][1]
        if value not in ("NaN", "+Inf", "-Inf"):
            return float(value)
    return None


def vllm_flags() -> dict[str, str]:
    """The two flags Phase 6 moves, read back out of the launch log.

    ponytail: a grep, because vLLM exposes neither in /metrics - cache_config_info
    covers the cache only, and api_server_count lives in the front end. Reports
    "unknown" rather than guessing. restart_vllm.sh truncates the log on every
    restart, so the last match in it describes the server now running.

    vLLM 0.10.2 dumps its config as a Python dict, so the name arrives quoted:
    `'max_num_seqs': 256`. It also logs "api_server_count more than 1; disabling
    stats", which reads alarming and is not - that is the periodic console
    logger, not the Prometheus exporter, as every Phase 2 number measured under
    --api-server-count 4 shows.
    """
    found = {"api_server_count": "unknown", "max_num_seqs": "unknown"}
    if not VLLM_LOG.exists():
        return found
    text = VLLM_LOG.read_text(errors="replace")
    for key in found:
        match = re.findall(rf"{key}'?\s*[=:]\s*'?(\d+)", text)
        if match:
            found[key] = match[-1]
    return found


def census(workers: int) -> dict[str, str]:
    """Everything needed to attribute a row in the iteration log to a config."""
    config = {"agent_workers": str(workers), "cpu_count": str(os.cpu_count()), **vllm_flags()}
    try:
        with urllib.request.urlopen(f"{AGENT}/health", timeout=5) as resp:
            health = json.load(resp)
        for key in ("max_iterations", "schema_values", "langfuse"):
            config[key] = health.get(key, "?")
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        for key in ("max_iterations", "schema_values", "langfuse"):
            config[key] = "agent not answering"
    return config


# Matched against each process's full cmdline, for the two groups the iteration
# log talks about. Everything else is reported by name, because the workers-2 run
# found 9 of 16 busy cores belonging to neither.
# The load generator is a single Python process holding thousands of pending
# aiohttp tasks, so it is part of the system under test and gets its own row.
# `spawn_main` is how a uvicorn --workers child appears: multiprocessing re-execs
# it with -c, so "uvicorn" is nowhere in its cmdline. Missing it is what left 9
# of 16 busy cores unattributed in the workers-2 row - and hid four leaked
# children burning 5.1 cores for 19 hours after it.
PROC_GROUPS = {
    "vllm": ("vllm serve", "VLLM::", "EngineCore"),
    "agent": ("uvicorn", "spawn_main"),
    "driver": ("driver.py",),
    "langfuse": ("clickhouse", "postgres", "langfuse", "minio", "redis"),
}
TOP_PROCESSES = 6


def cpu_snapshot() -> dict:
    """Every process's CPU seconds so far, plus the box's.

    ponytail: /proc arithmetic rather than a node_exporter - the question is only
    "who is eating the 16 cores". ps %cpu cannot answer it, since it averages over
    each process's whole lifetime and vLLM has been up for days. Reading every
    process rather than a fixed list is the lesson from the workers-2 run, where
    the two groups I thought to name accounted for 2.4 of 11.7 busy cores.
    """
    if not Path("/proc/stat").exists():
        return {}
    ticks = os.sysconf("SC_CLK_TCK")
    fields = Path("/proc/stat").read_text().split("\n")[0].split()[1:]
    idle = sum(int(f) for f in fields[3:5])  # idle + iowait
    snapshot = {
        "box_total": sum(int(f) for f in fields) / ticks,
        "box_busy": (sum(int(f) for f in fields) - idle) / ticks,
        "procs": {},
    }
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue  # processes come and go; a race here is not an error
        # utime and stime are fields 14 and 15, but comm can contain spaces, so
        # everything is offset from the last ')' rather than split from the left.
        values = stat[stat.rindex(")") + 2:].split()
        name = stat[stat.index("(") + 1:stat.rindex(")")]
        snapshot["procs"][entry.name] = (name, cmdline, (int(values[11]) + int(values[12])) / ticks)
    return snapshot


def cpu_report(before: dict, after: dict, wall: float) -> dict:
    """Mean cores busy over the run: the box, the two named groups, top consumers."""
    if not before or not after or wall <= 0:
        return {}
    deltas = {}
    for pid, (name, cmdline, seconds) in after["procs"].items():
        was = before["procs"].get(pid)
        # A pid absent beforehand started during the run, so all of its CPU counts.
        deltas[pid] = (name, cmdline, seconds - (was[2] if was else 0.0))

    groups = {
        group: round(sum(d / wall for _n, cmd, d in deltas.values()
                         if any(p in cmd for p in patterns)), 2)
        for group, patterns in PROC_GROUPS.items()
    }
    top = sorted(deltas.values(), key=lambda item: item[2], reverse=True)[:TOP_PROCESSES]
    return {
        "cores": round((after["box_total"] - before["box_total"]) / wall, 1),
        "box_busy_cores": round((after["box_busy"] - before["box_busy"]) / wall, 2),
        **{f"{group}_cores": value for group, value in groups.items()},
        "top": [{"name": name, "cores": round(d / wall, 2)} for name, _cmd, d in top if d > 0],
    }


def run_load(rps: float, duration: int, label: str, out: Path) -> bool:
    ends = time.strftime("%H:%M:%S", time.localtime(time.time() + duration))
    print(f"\n{'=' * 70}")
    print(f"  Load: {rps} agent runs/s for {duration}s, ending {ends}   label={label}")
    print(f"  {DASHBOARD_URL}?refresh=5s&from=now-5m&to=now")
    print(f"  {DIM}capture the window while this runs - grafana_before.png /")
    print(f"  grafana_after.png are the pair around whichever change moves it{OFF}")
    print(f"{'=' * 70}\n")
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--rps", str(rps), "--duration", str(duration),
         "--label", label, "--out", str(out)],
        cwd=ROOT, check=False,
    )
    return completed.returncode == 0


def diagnose(client: dict, config: dict) -> dict:
    """Client-side percentiles beside vLLM's own view of the same traffic."""
    server = {name: scalar(expr) for name, expr in QUERIES.items()}

    # Calls per agent run is measured, not assumed: it is 2 when verify accepts
    # and 5 after two revises, and the mix depends on the question pool.
    achieved = client["achieved_rps"]
    calls_per_run = (server["calls_per_second"] / achieved
                     if server["calls_per_second"] and achieved else None)
    # The whole point of the phase: what the agent adds on top of the model calls
    # it is waiting on. Compared at p50, because summing p95s is not a latency.
    overhead = (client["latency_p50"] - server["vllm_call_p50"] * calls_per_run
                if calls_per_run and server["vllm_call_p50"] else None)
    server["calls_per_run"] = calls_per_run
    server["agent_overhead_p50"] = overhead
    return server


def verdict(client: dict, server: dict) -> tuple[bool, list[str]]:
    """PASS needs the latency *and* the arrival rate *and* a visible window."""
    problems = []
    if client["latency_p95"] >= SLO_P95_SECONDS:
        problems.append(f"p95 {client['latency_p95']:.2f}s misses the {SLO_P95_SECONDS:.0f}s SLO "
                        f"by {client['latency_p95'] - SLO_P95_SECONDS:+.2f}s")
    failed = client["timeouts"] + client["http_errors"] + client["client_errors"]
    if failed:
        problems.append(f"{failed} of {client['total_requests']} requests failed")
    if client["achieved_rps"] < 0.95 * client["requested_rps"]:
        # Not the driver's fault: it fires open-loop and did offer the full rate.
        # achieved_rps is completions over a wall clock that includes the drain,
        # so a shortfall means the system needed longer than the run to finish
        # the run's arrivals - a backlog, by definition.
        problems.append(f"completed only {client['achieved_rps']:.2f}/s of the "
                        f"{client['requested_rps']:.1f}/s offered - it took "
                        f"{client['wall_clock_seconds']:.0f}s to finish "
                        f"{client['duration_seconds']}s of arrivals, so a backlog built up")
    scrape = server.get("scrape_success_percent")
    if scrape is not None and scrape < 95:
        problems.append(f"Prometheus captured only {scrape:.0f}% of scrapes - the dashboard has "
                        "gaps, so this window is not evidence of anything")
    return not problems, problems


def show(client: dict, server: dict, config: dict) -> None:
    def num(value, unit="", digits=2):
        return f"{value:.{digits}f}{unit}" if isinstance(value, (int, float)) else "n/a"

    print(f"\n{'=' * 70}")
    print(f"  {config['label']}  -  {client['requested_rps']:.0f} rps requested, "
          f"{client['duration_seconds']}s, {client['total_requests']} runs")
    print(f"  {DIM}agent workers {config['agent_workers']}, max_iterations "
          f"{config['max_iterations']}, langfuse {config['langfuse']}, "
          f"schema values {config['schema_values']}, "
          f"api-server-count {config['api_server_count']}, "
          f"max-num-seqs {config['max_num_seqs']}, {config['cpu_count']} cpus{OFF}")
    print(f"{'=' * 70}")
    print(f"  agent, client side    p50 {num(client['latency_p50'], 's')}  "
          f"p95 {num(client['latency_p95'], 's')}  p99 {num(client['latency_p99'], 's')}  "
          f"max {num(client['latency_max'], 's')}")
    print(f"  arrival rate          {num(client['achieved_rps'])} of "
          f"{client['requested_rps']:.1f} rps achieved")
    print(f"  vLLM, per call        p50 {num(server['vllm_call_p50'], 's')}  "
          f"p95 {num(server['vllm_call_p95'], 's')}  "
          f"{num(server['calls_per_second'])} calls/s "
          f"= {num(server['calls_per_run'])} per run")
    print(f"  where a call goes     queue {num(server['queue_mean'], 's')}  "
          f"prefill {num(server['prefill_mean'], 's')}  "
          f"decode {num(server['decode_mean'], 's')}  "
          f"unaccounted {num(server['unaccounted_mean'], 's')}")
    print(f"  {YELLOW}agent overhead{OFF}        "
          f"{num(server['agent_overhead_p50'], 's')} at p50, i.e. client p50 minus "
          f"{num(server['calls_per_run'], '', 1)} model calls")
    print(f"  engine               running {num(server['running'], '', 0)}  "
          f"waiting peak {num(server['waiting_peak'], '', 0)}  "
          f"batch p95 {num(server['batch_tokens_p95'], ' tok/step', 0)}  "
          f"preempt {num(server['preemptions_per_second'])}/s")
    print(f"  headroom             KV {num(server['kv_used'] and server['kv_used'] * 100, '%', 1)}  "
          f"prefix hits {num(server['prefix_hit_rate'] and server['prefix_hit_rate'] * 100, '%', 1)}  "
          f"output p95 {num(server['output_tokens_p95'], ' tok', 0)}  "
          f"truncated {num(server['finished_length_share'] and server['finished_length_share'] * 100, '%', 1)}")
    print(f"  scrape success       {num(server['scrape_success_percent'], '%', 1)}")
    cpu = server.get("cpu") or {}
    if cpu:
        print(f"  {YELLOW}cpu, mean cores{OFF}       box {num(cpu['box_busy_cores'], '', 1)} of "
              f"{num(cpu['cores'], '', 0)} busy   vLLM {num(cpu['vllm_cores'], '', 1)}   "
              f"agent {num(cpu['agent_cores'], '', 1)}   "
              f"driver {num(cpu['driver_cores'], '', 1)}   "
              f"langfuse {num(cpu['langfuse_cores'], '', 1)}")
        busiest = "  ".join(f"{p['name']} {p['cores']}" for p in cpu.get("top", []))
        print(f"  busiest processes     {DIM}{busiest}{OFF}")
    print(f"{'=' * 70}\n")


def failure_breakdown(payload: dict) -> list[str]:
    """The distinct client-side failures, commonest first."""
    counts = Counter(
        r.get("error") or r["status"]
        for r in payload.get("results", []) if r["status"] != "ok"
    )
    return [f"{count}x {error[:110]}" for error, count in counts.most_common(4)]


def record(row: dict) -> None:
    """Append one run, so the report's table is reproducible from the repo."""
    runs = json.loads(SLO_FILE.read_text()) if SLO_FILE.exists() else []
    runs.append(row)
    SLO_FILE.parent.mkdir(parents=True, exist_ok=True)
    SLO_FILE.write_text(json.dumps(runs, indent=2))
    ok(f"recorded run {len(runs)} in {_rel(SLO_FILE)}")


def slowest(label: str, count: int) -> int:
    """Per-node breakdown of the slowest runs of a load test, out of Langfuse.

    Grafana can only say whether the serving layer is innocent. When it is - the
    rps-3 run had vLLM at p95 1.13s, zero queueing and p99 36.9s at the client -
    the remaining time is inside a run, and the trace is the only thing that sees
    it. This is the two observability layers doing the job together.
    """
    from phase4 import check_langfuse, client

    # Spans nest - ChatOpenAI sits inside generate_sql, which sits inside the
    # root - so the aggregate counts graph nodes only. The per-trace list stays
    # unfiltered, which is what shows whether a node's time is its model call or
    # its own work.
    nodes = {"attach_schema", "generate_sql", "execute", "verify", "revise", "route_after_verify"}
    if not check_langfuse():
        return 1
    lf = client()
    # driver.py tags every request, so one load run is one tag.
    traces = lf.api.trace.list(tags=[f"label:{label}"], limit=100,
                               order_by="timestamp.desc").data
    timed = [t for t in traces if t.latency]
    if not timed:
        bad(f"no traces tagged label:{label} - was the run driven by load_test/driver.py?")
        return 1

    ranked = sorted(timed, key=lambda t: t.latency, reverse=True)
    print(f"\n{len(timed)} traces tagged label:{label}, "
          f"latency p50 {sorted(t.latency for t in timed)[len(timed) // 2]:.2f}s, "
          f"slowest {ranked[0].latency:.2f}s\n")

    totals: dict[str, float] = {}
    for trace in ranked[:count]:
        spans = lf.api.trace.get(trace.id).observations
        durations = [
            (o.name, (o.end_time - o.start_time).total_seconds())
            for o in spans if o.end_time and o.start_time and o.parent_observation_id
        ]
        print(f"  {trace.latency:7.2f}s  {trace.id[:12]}")
        for name, seconds in sorted(durations, key=lambda d: d[1], reverse=True)[:6]:
            share = 100 * seconds / trace.latency if trace.latency else 0
            print(f"    {seconds:7.2f}s  {share:4.0f}%  {name}")
        for name, seconds in durations:
            if name in nodes:
                totals[name] = totals.get(name, 0.0) + seconds

    print(f"\n  across the {count} slowest runs, by graph node:")
    everything = sum(totals.values()) or 1.0
    for name, seconds in sorted(totals.items(), key=lambda t: t[1], reverse=True):
        print(f"    {seconds:8.2f}s  {100 * seconds / everything:4.0f}%  {name}")
    return 0


def report() -> int:
    if not SLO_FILE.exists():
        warn(f"{_rel(SLO_FILE)} does not exist yet - no runs recorded")
        return 0
    runs = json.loads(SLO_FILE.read_text())
    print(f"\n{len(runs)} recorded run(s)\n")
    print(f"  {'label':22} {'rps':>5} {'p50':>7} {'p95':>7} {'calls':>6} {'ovhd':>7} "
          f"{'wrk':>4} {'iter':>5}  verdict")
    for r in runs:
        c, s, cfg = r["client"], r["server"], r["config"]
        def num(v, digits=2):
            return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "n/a"
        mark = f"{GREEN}PASS{OFF}" if r["passed"] else f"{RED}FAIL{OFF}"
        print(f"  {cfg['label'][:22]:22} {c['achieved_rps']:5.1f} {num(c['latency_p50']):>7} "
              f"{num(c['latency_p95']):>7} {num(s.get('calls_per_run'), 1):>6} "
              f"{num(s.get('agent_overhead_p50')):>7} {cfg['agent_workers']:>4} "
              f"{cfg['max_iterations']:>5}  {mark}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rps", type=float, default=SLO_RPS, help=f"agent runs/s (default: {SLO_RPS:.0f}, the SLO)")
    parser.add_argument("--duration", type=int, default=300, help="seconds of load (default: 300, the SLO window)")
    parser.add_argument("--label", default="baseline", help="names this configuration, in the log and in Langfuse")
    parser.add_argument("--workers", type=int, default=1, help="uvicorn workers, if this script starts the agent")
    parser.add_argument("--verify-only", action="store_true", help="stack and config census, fire nothing")
    parser.add_argument("--report-only", action="store_true", help="re-read the recorded runs")
    parser.add_argument("--slowest", type=int, metavar="N",
                        help="skip the load: break the N slowest traces of --label down by node")
    args = parser.parse_args()

    print(f"Phase 6 - SLO: p95 < {SLO_P95_SECONDS:.0f}s at {SLO_RPS:.0f}+ agent runs/s over 5 minutes")
    if args.report_only:
        return report()
    if args.slowest:
        return slowest(args.label, args.slowest)

    if not check_services():  # Prometheus, Grafana, and the vllm scrape target
        return 1
    print("\nvLLM and agent")
    if not alive("http://localhost:8000"):
        bad("vLLM unreachable at http://localhost:8000 - run: ./scripts/start_vllm.sh")
        return 1
    ok("vLLM reachable at http://localhost:8000")
    agent = start_agent(workers=args.workers)

    config = census(args.workers) | {"label": args.label}
    print(f"  {DIM}workers {config['agent_workers']}, max_iterations {config['max_iterations']}, "
          f"langfuse {config['langfuse']}, schema values {config['schema_values']}, api-server-count "
          f"{config['api_server_count']}, max-num-seqs {config['max_num_seqs']}, "
          f"{config['cpu_count']} cpus{OFF}")
    if agent.proc is None and args.workers > 1:
        warn(f"--workers {args.workers} was ignored: an agent was already running, so this run "
             "would be attributed to the wrong configuration. Stop it and re-run.")

    if args.verify_only:
        print(f"\n{GREEN}Stack ready.{OFF} Config census above.")
        return 0

    out = ROOT / "results" / f"phase6_load_{args.label}.json"
    client = server = None
    cpu_before = cpu_snapshot()
    # Grafana has no image renderer here and its admin password is not the
    # provisioned one, so the before/after PNGs are browser captures. Recording
    # the absolute window makes them reproducible after the fact instead of
    # depending on someone watching the right five minutes.
    started = time.time()
    try:
        if not run_load(args.rps, args.duration, args.label, out) or not out.exists():
            print(f"\n{RED}driver.py failed.{OFF} See the output above.")
            return 1
        # Analysis before shutdown: the [5m] PromQL window has to still be the
        # load window, and a crash in here must not throw the run away.
        payload = json.loads(out.read_text())
        client = payload["summary"]
        server = diagnose(client, config)
        server["cpu"] = cpu_report(cpu_before, cpu_snapshot(), client["wall_clock_seconds"])
        failures = failure_breakdown(payload)
    finally:
        ended = time.time()
        stop_agent(agent)

    show(client, server, config)
    passed, problems = verdict(client, server)
    window = f"{DASHBOARD_URL}?from={int(started * 1000)}&to={int(ended * 1000)}"
    print(f"  screenshot this exact window: {window}\n")
    record({"config": config, "client": client, "server": server, "failures": failures,
            "passed": passed, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "grafana": window})

    for problem in problems:
        bad(problem)
    for failure in failures:
        warn(f"client-side failure: {failure}")
    if passed:
        ok(f"SLO held: p95 {client['latency_p95']:.2f}s at {client['achieved_rps']:.1f} runs/s")
        if args.rps <= SLO_RPS:
            print(f"  {DIM}now push past it (--rps 15, 20, 25) - a green check at the SLO point")
            print(f"  says nothing about where the cliff is{OFF}")
    else:
        # The single most useful number when the SLO misses: how much of the
        # client's latency vLLM never saw.
        overhead = server.get("agent_overhead_p50")
        if overhead is not None and client["latency_p50"]:
            share = 100 * overhead / client["latency_p50"]
            print(f"  {YELLOW}{share:.0f}% of the client's p50 is outside vLLM{OFF} - "
                  f"{overhead:.2f}s of {client['latency_p50']:.2f}s")
    # A missed SLO is a finding to diagnose, not a broken script, so the exit code
    # tracks whether the *measurement* worked.
    return 0


if __name__ == "__main__":
    sys.exit(main())
