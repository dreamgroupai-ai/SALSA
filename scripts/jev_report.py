#!/usr/bin/env python3
"""Score a JevBench run of the SALSA decision API and put it next to Jev.

Reads ``<run>/results.jsonl`` (written by ``jevbench.cli run``) and, when present,
``<run>/load.json`` (``scripts/jev_loadtest.py``), and writes ``report.json`` and
``report.md`` into the run directory.

Only the public half of JevBench is available (easy 48, standard 72, hard 111;
the 146-item judge tier is not public at all), so everything here is a
**public-subset proxy** built with the benchmark's own v1.3 scoring functions.
It is not an official JevBench Score, which also needs the held-out items.

Head-to-head numbers compare systems on *identical* items: the benchmark
publishes per-item outcomes of every system for the public items
(``results/v1.2/jevbench-v1.2-per-task.json``).  Significance is an exact
McNemar test on the discordant items.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

TIER_BY_FILE = {"easy.jsonl": "easy", "original.jsonl": "standard", "hard.jsonl": "hard"}
TIERS = ("easy", "standard", "hard")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def num(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tasks", required=True, help="the comma-separated jsonl list the run used")
    ap.add_argument("--jevbench-dir", default="datasets/jevbench")
    ap.add_argument("--compare", default="jev-1.13.0,simplejev-qwen3.8-27b",
                    help="system keys from the published per-task file")
    ap.add_argument("--endpoint-kind", default="gpu",
                    help="'gpu' applies JevBench's self-host latency rule (x2 + 0.15 s); 'api' none")
    # The benchmark prices a self-hosted model without a tariff at its base model's
    # hosted reference price.  0.09 $/M input, 0 output is the reference it used for
    # Qwen3.8-27B served as a direct-logit classifier
    # (results/v1.2/additions/simplejev-qwen3.8-27b.json, cost.basis_final).
    ap.add_argument("--ref-price-in-per-m", type=float, default=0.09)
    ap.add_argument("--ref-price-out-per-m", type=float, default=0.0)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.jevbench_dir))
    from jevbench import composite_v13 as v13
    from jevbench.metrics import ece_top_label, percentile
    from jevbench.tasks import load_jsonl

    tasks, tier_of = [], {}
    for part in [p.strip() for p in args.tasks.split(",") if p.strip()]:
        for t in load_jsonl(part):
            tasks.append(t)
            tier_of[t.id] = TIER_BY_FILE.get(os.path.basename(part), "other")
    by_id = {t.id: t for t in tasks}
    recs = {r["task_id"]: r for r in load_records(os.path.join(args.run_dir, "results.jsonl"))}
    manifest_path = os.path.join(args.run_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    # ---------------- accuracy, per tier and per family ---------------- #
    tiers = {}
    for tier in TIERS:
        ids = [t.id for t in tasks if tier_of[t.id] == tier]
        if not ids:
            continue
        correct = sum(bool(recs.get(i, {}).get("correct")) for i in ids)
        tiers[tier] = {
            "n": len(ids),
            "attempted": sum(i in recs for i in ids),
            "failed": sum(i in recs and not recs[i]["ok"] for i in ids),
            "correct": correct,
            "accuracy": correct / len(ids),   # failures and unattempted count wrong, as on the board
            "chance": sum(1 / len(by_id[i].labels) for i in ids) / len(ids),
        }
    families = defaultdict(lambda: [0, 0])
    for t in tasks:
        if tier_of[t.id] == "hard":
            families[t.family][0] += bool(recs.get(t.id, {}).get("correct"))
            families[t.family][1] += 1

    # ---------------- calibration (hard tier, as on the board) ---------------- #
    hard = [recs[t.id] for t in tasks if tier_of[t.id] == "hard" and recs.get(t.id, {}).get("probs")]
    pairs = [(max(r["probs"].values()), bool(r["correct"])) for r in hard]
    ece = ece_top_label(pairs)["ece"] if pairs else None
    tvds = [v13.tvd(recs[t.id]["probs"], t.provenance["gold_probs"], t.labels)
            for t in tasks if t.provenance.get("gold_probs") and recs.get(t.id, {}).get("probs")]
    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    brier = [sum((r["probs"][k] - (k == str(by_id[r["task_id"]].expected))) ** 2 for k in r["probs"]) for r in hard]

    # ---------------- speed: serial run, standard tier (judge is not public) ---------------- #
    std_lat = [recs[t.id]["latency_s"] for t in tasks if tier_of[t.id] == "standard" and t.id in recs]
    all_lat = [r["latency_s"] for r in recs.values()]
    p50, p95 = percentile(std_lat, 0.5), percentile(std_lat, 0.95)

    # Diagnostics from the gateway's per-response ``runtime``: how much probability the
    # model puts on the class tokens at all (format adherence), and how much of the
    # serial latency is vLLM itself rather than gateway + HTTP.
    rt = [r.get("runtime") or {} for r in recs.values()]
    mass = sorted(m for x in rt for m in (x.get("class_mass") or {}).values())
    engine_ms = [x["timing_ms"]["engine"] for x in rt if (x.get("timing_ms") or {}).get("engine") is not None]

    # ---------------- cost estimate at the reference price ---------------- #
    tok_in = [r["usage"].get("input_tokens") for r in recs.values() if r.get("usage")]
    tok_out = [r["usage"].get("output_tokens") or 0 for r in recs.values() if r.get("usage")]
    mean_in = sum(tok_in) / len(tok_in) if tok_in and None not in tok_in else None
    mean_out = sum(tok_out) / len(tok_out) if tok_out else 0.0
    usd_1000 = (mean_in * args.ref_price_in_per_m + mean_out * args.ref_price_out_per_m) / 1e6 * 1000 \
        if mean_in is not None else None

    axes = {
        "intelligence": v13.intelligence({k: v["accuracy"] for k, v in tiers.items()}),
        "calibration": v13.calibration(ece, mean_tvd),
        "speed": v13.speed(p50, p95, args.endpoint_kind),
        "cost": v13.cost(usd_1000) if usd_1000 else None,
    }
    proxy_score = v13.jevbench_score(axes) if None not in axes.values() else None

    # ---------------- head-to-head on identical public items ---------------- #
    per_task = json.load(open(os.path.join(args.jevbench_dir, "results/v1.2/jevbench-v1.2-per-task.json")))
    published = {row["key"]: row for row in
                 json.load(open(os.path.join(args.jevbench_dir, "results/v1.2/jevbench-v1.2-results.json")))["systems"]}
    h2h = {}
    for key in [k.strip() for k in args.compare.split(",") if k.strip()]:
        sysrow = per_task["systems"].get(key)
        if sysrow is None:
            print(f"[report] unknown system {key!r}; skipped", file=sys.stderr)
            continue
        theirs = sysrow["public_tasks"]
        out = {}
        for tier in (*TIERS, "all"):
            ids = [t.id for t in tasks if (tier == "all" or tier_of[t.id] == tier)
                   and theirs.get(t.id, ["n"])[0] != "n"]
            if not ids:
                continue
            ours_ok = {i: bool(recs.get(i, {}).get("correct")) for i in ids}
            their_ok = {i: theirs[i][0] == "c" for i in ids}
            b = sum(ours_ok[i] and not their_ok[i] for i in ids)
            c = sum(their_ok[i] and not ours_ok[i] for i in ids)
            out[tier] = {"n": len(ids), "salsa": sum(ours_ok.values()) / len(ids),
                         "them": sum(their_ok.values()) / len(ids),
                         "salsa_only": b, "them_only": c, "mcnemar_p": mcnemar_exact(b, c)}
        pub = published.get(key, {})
        h2h[key] = {"display": sysrow["display"], "same_items": out,
                    "published_all_items": {"tiers": pub.get("tiers"), "axes": pub.get("axes"),
                                            "jevbench_score_v13": pub.get("jevbench_score"),
                                            "speed": pub.get("speed"), "endpoint_kind": pub.get("endpoint_kind")}}

    load_path = os.path.join(args.run_dir, "load.json")
    load = json.load(open(load_path)) if os.path.exists(load_path) else None

    report = {
        "kind": "public-subset proxy (not an official JevBench Score)",
        "run_dir": os.path.abspath(args.run_dir),
        "manifest": manifest,
        "coverage": {"tasks": len(tasks), "attempted": len(recs),
                     "ok": sum(r["ok"] for r in recs.values()),
                     "valid": sum(bool(r.get("valid")) for r in recs.values()),
                     "renormalized": sum(bool(r.get("renormalized")) for r in recs.values()),
                     "status_codes": dict(sorted(
                         {str(k): sum(r.get("status_code") == k for r in recs.values())
                          for k in {r.get("status_code") for r in recs.values()}}.items()))},
        "tiers": tiers,
        "hard_families": {f: {"correct": c, "n": n, "accuracy": c / n} for f, (c, n) in sorted(families.items())},
        "calibration": {"ece_hard": ece, "mean_tvd_probability_items": mean_tvd, "n_tvd": len(tvds),
                        "brier_hard": sum(brier) / len(brier) if brier else None},
        "class_mass": {"mean": sum(mass) / len(mass) if mass else None, "p05": percentile(mass, 0.05),
                       "min": mass[0] if mass else None, "n": len(mass)},
        "latency_serial": {"standard_p50_s": p50, "standard_p95_s": p95,
                           "all_p50_s": percentile(all_lat, 0.5), "all_p95_s": percentile(all_lat, 0.95),
                           "vllm_ms_p50": percentile(engine_ms, 0.5), "vllm_ms_p95": percentile(engine_ms, 0.95),
                           "endpoint_kind": args.endpoint_kind,
                           "adjusted_p50_s": v13.adjusted_latency(p50, args.endpoint_kind),
                           "adjusted_p95_s": v13.adjusted_latency(p95, args.endpoint_kind)},
        "cost_estimate": {"mean_input_tokens": mean_in, "mean_output_tokens": mean_out,
                          "usd_per_1000_decisions": usd_1000,
                          "basis": f"ESTIMATE: reference price ${args.ref_price_in_per_m}/M in, "
                                   f"${args.ref_price_out_per_m}/M out x measured tokens"},
        "axes_proxy": axes,
        "jevbench_score_v13_proxy": proxy_score,
        "head_to_head": h2h,
        "load": load,
    }
    with open(os.path.join(args.run_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # ---------------- markdown ---------------- #
    md = [f"# SALSA on JevBench (public items) -- {manifest.get('run_label', os.path.basename(args.run_dir))}", "",
          "Public-subset proxy built with JevBench's own v1.3 functions; not an official score "
          "(no judge tier, no held-out items).", ""]
    cov = report["coverage"]
    md += [f"Coverage: {cov['attempted']}/{cov['tasks']} attempted, {cov['ok']} ok, {cov['valid']} valid, "
           f"{cov['renormalized']} renormalised; HTTP {cov['status_codes']}", ""]
    md += ["| tier | n | accuracy | chance |", "|---|---:|---:|---:|"]
    md += [f"| {k} | {v['n']} | {pct(v['accuracy'])} | {pct(v['chance'])} |" for k, v in tiers.items()]
    md += ["", "| proxy axis | value |", "|---|---:|"]
    md += [f"| {k} | {num(v, '.1f')} |" for k, v in axes.items()]
    md += [f"| **v1.3 composite (proxy)** | **{num(proxy_score, '.1f')}** |", ""]
    md += [f"ECE (hard) {num(ece, '.3f')} - mean TVD to gold on {len(tvds)} probability items {num(mean_tvd, '.3f')} - "
           f"serial latency (standard) p50 {num(p50, '.3f')} s / p95 {num(p95, '.3f')} s, "
           f"adjusted x2+0.15 s for '{args.endpoint_kind}' - est. ${num(usd_1000, '.4f')} per 1,000 decisions "
           f"({num(mean_in, '.0f')} input tokens/decision)", "",
           f"Probability mass on the class tokens (format adherence): mean {num(report['class_mass']['mean'], '.3f')}, "
           f"p05 {num(report['class_mass']['p05'], '.3f')}, min {num(report['class_mass']['min'], '.3f')} - "
           f"vLLM share of serial latency: p50 {num(report['latency_serial']['vllm_ms_p50'], '.1f')} ms, "
           f"p95 {num(report['latency_serial']['vllm_ms_p95'], '.1f')} ms", ""]
    for key, h in h2h.items():
        md += [f"## vs {h['display']} -- identical public items", "",
               "| tier | n | SALSA | them | SALSA only | them only | McNemar p |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {t} | {v['n']} | {pct(v['salsa'])} | {pct(v['them'])} | {v['salsa_only']} | "
               f"{v['them_only']} | {num(v['mcnemar_p'], '.3f')} |" for t, v in h["same_items"].items()]
        pub = h["published_all_items"]
        if pub.get("axes"):
            sp = pub.get("speed") or {}
            md += ["", f"Published (all items incl. held-out): v1.3 score {num(pub['jevbench_score_v13'], '.1f')}; "
                   f"axes {', '.join(f'{k} {v:.1f}' for k, v in pub['axes'].items())}; "
                   f"raw p50 {num(sp.get('p50_s_raw'), '.3f')} s over the internet "
                   f"({pub.get('endpoint_kind')})."]
        md.append("")
    if load:
        md += ["## Throughput and latency under load (closed loop)", "",
               "| concurrency | decisions/s | input tok/s | p50 s | p95 s | p99 s | p50 / serial |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {s['concurrency']} | {num(s['decisions_per_s'])} | {num(s['input_tokens_per_s'], '.0f')} | "
               f"{num(s['latency_s']['p50'], '.3f')} | {num(s['latency_s']['p95'], '.3f')} | "
               f"{num(s['latency_s']['p99'], '.3f')} | {num(s.get('p50_vs_serial'))} |" for s in load["levels"]]
        md.append("")
    with open(os.path.join(args.run_dir, "report.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
