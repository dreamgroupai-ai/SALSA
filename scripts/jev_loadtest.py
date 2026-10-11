#!/usr/bin/env python3
"""Throughput and latency of a /v1/systemone decision endpoint under load.

JevBench measures latency serially (one request in flight), which is the
official Speed axis.  It then multiplies self-hosted latency by 2 (+0.15 s) to
*approximate* production load -- its README calls that "an assumption, not a
measurement".  This script is that measurement: a closed-loop concurrency sweep
replaying the benchmark's own request bodies (built by its ``typesafe``
adapter), reporting at every level

  * throughput: decisions/s and prompt tokens/s;
  * client latency: mean, p50, p90, p95, p99, max;
  * server split: vLLM time vs. gateway overhead (from the response ``runtime``);
  * the JevBench Speed axis at that latency, raw and with the x2 + 0.15 s rule;
  * vLLM's own histograms (queue, prefill, TTFT, end-to-end) as deltas of its
    Prometheus counters, when ``--vllm-metrics`` is given.

Stdlib only; one keep-alive connection per worker thread.

    python scripts/jev_loadtest.py --endpoint http://127.0.0.1:8000 \\
        --tasks datasets/jevbench/datasets/public/easy.jsonl,... \\
        --concurrency 1,4,16,64 --vllm-metrics http://127.0.0.1:8100/metrics \\
        --out logs/jev/run/load.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

# vLLM histogram families whose (sum, count) deltas give a per-level mean.
VLLM_HISTOGRAMS = {
    "e2e_s": "vllm:e2e_request_latency_seconds",
    "queue_s": "vllm:request_queue_time_seconds",
    "prefill_s": "vllm:request_prefill_time_seconds",
    "ttft_s": "vllm:time_to_first_token_seconds",
}
VLLM_COUNTERS = {"prompt_tokens": "vllm:prompt_tokens_total"}


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def load_bodies(jevbench_dir, task_spec, model):
    sys.path.insert(0, os.path.abspath(jevbench_dir))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.tasks import load_jsonl

    adapter = TypeSafeAdapter(endpoint="http://unused", model=model)
    tasks = [t for part in task_spec.split(",") if part.strip() for t in load_jsonl(part.strip())]
    return [json.dumps(adapter.build_request(t)).encode() for t in tasks]


def jevbench_speed(jevbench_dir, p50, p95):
    sys.path.insert(0, os.path.abspath(jevbench_dir))
    from jevbench import composite_v13 as c

    if p50 is None or p95 is None:
        return None, None
    return c.speed(p50, p95, "api"), c.speed(p50, p95, "gpu")


def scrape(url):
    """Sum every sample of each wanted family across label sets (DP ranks, models)."""
    if not url:
        return None
    try:
        text = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception as e:  # metrics are a diagnostic, never a reason to fail the run
        print(f"[load] metrics scrape failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def metrics_delta(before, after):
    if before is None or after is None:
        return None
    d = lambda k: after.get(k, 0.0) - before.get(k, 0.0)
    res = {}
    for name, fam in VLLM_HISTOGRAMS.items():
        n = d(fam + "_count")
        res[name + "_mean"] = d(fam + "_sum") / n if n > 0 else None
    for name, fam in VLLM_COUNTERS.items():
        res[name] = d(fam) if fam in after else None
    return res


class Worker(threading.Thread):
    def __init__(self, url, bodies, next_index, records, lock, timeout, headers):
        super().__init__(daemon=True)
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.path = u.hostname, u.port or 80, u.path or "/"
        self.bodies, self.next_index, self.records, self.lock = bodies, next_index, records, lock
        self.timeout, self.headers = timeout, headers
        self.conn = None

    def _post(self, body):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            self.conn.request("POST", self.path, body=body, headers=self.headers)
            resp = self.conn.getresponse()
            return resp.status, resp.read()
        except Exception:
            self.conn.close()
            self.conn = None
            raise

    def run(self):
        while True:
            i = self.next_index()
            if i is None:
                return
            body = self.bodies[i % len(self.bodies)]
            t0 = time.perf_counter()
            try:
                status, payload = self._post(body)
                err = None
            except Exception as e:
                status, payload, err = None, b"", f"{type(e).__name__}: {e}"
            t1 = time.perf_counter()
            rec = {"t0": t0, "t1": t1, "status": status, "error": err}
            if status == 200:
                try:
                    p = json.loads(payload)
                    rec["input_tokens"] = (p.get("usage") or {}).get("input_tokens")
                    rec["timing_ms"] = (p.get("runtime") or {}).get("timing_ms") or {}
                    if not p.get("answers"):
                        rec["error"] = "no answers"
                except ValueError:
                    rec["error"] = "invalid JSON"
            elif err is None:
                rec["error"] = f"HTTP {status}"
            with self.lock:
                self.records.append(rec)


def run_level(url, bodies, concurrency, n_requests, timeout, headers, seed):
    order = list(range(len(bodies)))
    random.Random(seed).shuffle(order)
    seq = iter(order[i % len(order)] for i in range(n_requests))
    lock = threading.Lock()

    def next_index():
        with lock:
            return next(seq, None)

    records = []
    workers = [Worker(url, bodies, next_index, records, lock, timeout, headers) for _ in range(concurrency)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return records


def summarize(records, concurrency, jevbench_dir):
    ok = [r for r in records if r["error"] is None]
    lat = [r["t1"] - r["t0"] for r in ok]
    wall = (max(r["t1"] for r in records) - min(r["t0"] for r in records)) if records else None
    tokens = sum(r.get("input_tokens") or 0 for r in ok)
    engine = [r["timing_ms"].get("engine") for r in ok if r.get("timing_ms", {}).get("engine") is not None]
    overhead = [r["timing_ms"]["total"] - r["timing_ms"]["engine"] for r in ok
                if r.get("timing_ms", {}).get("total") is not None and r["timing_ms"].get("engine") is not None]
    p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
    speed_raw, speed_adj = jevbench_speed(jevbench_dir, p50, p95)
    errors = {}
    for r in records:
        if r["error"] is not None:
            errors[r["error"][:80]] = errors.get(r["error"][:80], 0) + 1
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "ok": len(ok),
        "errors": errors,
        "wall_s": wall,
        "decisions_per_s": len(ok) / wall if wall else None,
        "input_tokens_per_s": tokens / wall if wall else None,
        "mean_input_tokens": tokens / len(ok) if ok else None,
        "latency_s": {
            "mean": sum(lat) / len(lat) if lat else None,
            "p50": p50,
            "p90": percentile(lat, 0.9),
            "p95": p95,
            "p99": percentile(lat, 0.99),
            "max": max(lat) if lat else None,
        },
        "server_engine_ms_p50": percentile(engine, 0.5),
        "server_engine_ms_p95": percentile(engine, 0.95),
        "gateway_overhead_ms_p50": percentile(overhead, 0.5),
        "jevbench_speed_raw": speed_raw,
        "jevbench_speed_selfhost_adjusted": speed_adj,
    }


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="base URL of the decision API")
    ap.add_argument("--tasks", required=True, help="comma-separated JevBench jsonl files")
    ap.add_argument("--jevbench-dir", default="datasets/jevbench")
    ap.add_argument("--model", default="salsa", help="'model' field sent in each request")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--requests", type=int, default=0,
                    help="requests per level; 0 = max(#tasks, 8 x concurrency)")
    ap.add_argument("--warmup", type=int, default=8, help="untimed requests before the sweep")
    ap.add_argument("--warmup-only", action="store_true", help="send the warm-up requests and stop")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--vllm-metrics", default=None, help="vLLM /metrics URL (optional)")
    ap.add_argument("--gpus", type=int, default=None, help="GPUs behind the endpoint, for cost per decision")
    ap.add_argument("--gpu-hour-usd", type=float, default=None,
                    help="price of one GPU-hour; unset -> no cost figure (never invented)")
    ap.add_argument("--api-key-env", default="", help="env var with a bearer token; empty = no auth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    url = args.endpoint.rstrip("/") + "/v1/systemone"
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if args.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"
    bodies = load_bodies(args.jevbench_dir, args.tasks, args.model)
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    print(f"[load] {len(bodies)} distinct requests, endpoint {url}"
          + ("" if args.warmup_only else f", levels {levels}"))

    if args.warmup:
        warm = run_level(url, bodies, 1, args.warmup, args.timeout_s, headers, args.seed + 1)
        bad = [r for r in warm if r["error"]]
        if bad:
            print(f"[load] warm-up failed: {bad[0]['error']}", file=sys.stderr)
            return 3
        print(f"[load] warm-up: {len(warm)} decisions ok")
    if args.warmup_only:
        return 0

    results = []
    print(f"{'conc':>5} {'ok':>6} {'err':>4} {'dec/s':>8} {'tok/s':>9} {'p50 s':>7} {'p95 s':>7} "
          f"{'p99 s':>7} {'vllm ms':>8} {'gw ms':>6} {'speed':>6} {'adj':>6}")
    for c in levels:
        n = args.requests or max(len(bodies), 8 * c)
        before = scrape(args.vllm_metrics)
        recs = run_level(url, bodies, c, n, args.timeout_s, headers, args.seed)
        s = summarize(recs, c, args.jevbench_dir)
        s["vllm"] = metrics_delta(before, scrape(args.vllm_metrics))
        if args.gpu_hour_usd is not None and args.gpus and s["decisions_per_s"]:
            s["usd_per_1000_decisions_gpu_time"] = (
                args.gpu_hour_usd * args.gpus / (s["decisions_per_s"] * 3600) * 1000)
        results.append(s)
        L = s["latency_s"]
        print(f"{c:>5} {s['ok']:>6} {sum(s['errors'].values()):>4} {fmt(s['decisions_per_s'], '8.2f')} "
              f"{fmt(s['input_tokens_per_s'], '9.0f')} {fmt(L['p50'], '7.3f')} {fmt(L['p95'], '7.3f')} "
              f"{fmt(L['p99'], '7.3f')} {fmt(s['server_engine_ms_p50'], '8.1f')} "
              f"{fmt(s['gateway_overhead_ms_p50'], '6.1f')} {fmt(s['jevbench_speed_raw'], '6.1f')} "
              f"{fmt(s['jevbench_speed_selfhost_adjusted'], '6.1f')}", flush=True)

    base = results[0]["latency_s"]["p50"] if results and results[0]["concurrency"] == 1 else None
    for s in results:
        s["p50_vs_serial"] = s["latency_s"]["p50"] / base if base and s["latency_s"]["p50"] else None
    peak = max(results, key=lambda s: s["decisions_per_s"] or 0) if results else None
    report = {
        "endpoint": args.endpoint,
        "n_distinct_requests": len(bodies),
        "tasks": args.tasks,
        "levels": results,
        "peak": {"concurrency": peak["concurrency"], "decisions_per_s": peak["decisions_per_s"]} if peak else None,
        "gpus": args.gpus,
        "gpu_hour_usd": args.gpu_hour_usd,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "closed loop: each level keeps exactly `concurrency` requests in flight",
    }
    if peak:
        print(f"[load] peak {peak['decisions_per_s']:.2f} decisions/s at concurrency {peak['concurrency']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0 if all(not s["errors"] for s in results) else 4


if __name__ == "__main__":
    raise SystemExit(main())
