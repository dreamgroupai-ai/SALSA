#!/usr/bin/env python3
"""Comparison tables for SALSA runs on JevBench public, in the layouts of LLM2Jev and AnyJev.

    python scripts/jev_compare.py \\
        --run "SALSA (Qwen3.8-27B)|vLLM 0.31, 1x B200|logs/jev/flow_vllm_...[|console.log]" ... \\
        --out logs/jev/compare_tables.md --json logs/jev/compare.json

Every number for our runs is computed from the run directory (results.jsonl,
manifest.json, load.json, orderflip.json, timing.json).  Published systems are scored
from JevBench's own per-item outcomes (results/v1.2/jevbench-v1.2-per-task.json),
so all rows use one scorer.  Numbers that only exist in the two papers are labelled
"as reported".

Protocols:
  * LLM2Jev (github.com/Yinsongxu/LLM2Jev, docs/jevbench.md): all 231 public items
    pooled; accuracy; serial latency P50/P95 over every decision.
  * AnyJev (arXiv 2610.00831, Tables 2 and 9): the 213 items whose expected answer is
    an option -- the 18 `score` items are skipped, which reproduces its 48/60/105 tier
    counts; ECE over 15 equal-mass bins of top-label confidence; paired bootstrap
    95% CIs over items (2,000 resamples); coverage at 5% error (cov@5%); order-flip
    rate under reversed options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from datetime import datetime

TIER_FILES = (("easy", "easy.jsonl"), ("original", "original.jsonl"), ("hard", "hard.jsonl"))

# Published systems scored from JevBench's per-item outcomes: LLM2Jev's table rows, the
# other systems built on Qwen3.8-27B, a Qwen3-32B zero-shot reader (AnyJev's largest
# model), and DeepSeek V4.1 Flash with thinking.
BASELINES = [
    ("jev-1.13.0", "Jev 1.13.0 (TypeSafe AI)"),
    ("simplejev-qwen3.8-27b", "SimpleJev (Qwen3.8-27B, direct logits)"),
    ("litjev", "LitJev (Qwen3.8-27B)"),
    ("reflex-27b", "reflex-27b (Qwen3.8-27B)"),
    ("jqv", "jqv (Qwen3-32B zero-shot)"),
    ("deepseek-flash", "DeepSeek V4.1 Flash (thinking, API)"),
    ("jev-local", "jev-local (Qwen3.5-9B)"),
    ("kev-8b", "kev 8B (research preview)"),
    ("kev-4b", "kev 4B (research preview)"),
    ("open-jev-zefan-2b", "Open-Jev 2B (Zefan Cai)"),
    ("jeff", "jeff (Logan Markewich, GLiFormer 400M)"),
    ("smalljev", "smalljev semantic-v9"),
    ("laya", "Laya (Convai Innovations, ModernBERT-large 421M)"),
]

# Only in the papers (cannot be recomputed here).
LLM2JEV_REPORTED = {"jev-1.13.0": 85.1, "jev-local": 74.9, "kev-8b": 71.4, "kev-4b": 66.2,
                    "open-jev-zefan-2b": 64.5, "jeff": 62.8, "smalljev": 60.6, "laya": 58.4}
LLM2JEV_OWN = ("LLM2Jev (Qwen3.5-4B), as reported", 76.2, 0.048, 0.346)
ANYJEV_TABLE2 = [  # model, answer mass, raw, debiased, ECE raw, ECE debiased, ECE + temp.
    ("Qwen3-32B", 1.000, 0.798, 0.789, 0.140, 0.128, 0.091),
    ("Qwen3-8B", 1.000, 0.704, 0.714, 0.278, 0.263, 0.114),
    ("gpt-oss-20b", 0.999, 0.704, 0.714, 0.140, 0.154, 0.102),
    ("Granite-3.3-8B", 0.999, 0.690, 0.700, 0.280, 0.270, 0.301),
    ("Qwen2.5-7B", 0.996, 0.676, 0.685, 0.259, 0.243, 0.085),
    ("Mistral-7B", 0.999, 0.615, 0.624, 0.322, 0.333, 0.104),
    ("Llama-3.2-3B", 0.999, 0.563, 0.559, 0.174, 0.163, 0.107),
    ("Qwen3-1.7B", 1.000, 0.549, 0.568, 0.395, 0.380, 0.105),
]
ANYJEV_TABLE9 = [("Qwen3-32B (AnyJev raw)", 1.000, 0.967, 0.590)]


def ece_equal_mass(pairs, bins=15):
    """ECE over `bins` equal-count bins of top-label confidence (AnyJev's definition)."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    n, total = len(pairs), 0.0
    for b in range(bins):
        chunk = pairs[b * n // bins:(b + 1) * n // bins]
        if chunk:
            total += len(chunk) / n * abs(sum(c for _, c in chunk) / len(chunk) - sum(p for p, _ in chunk) / len(chunk))
    return total


def cov_at(pairs, max_err=0.05):
    """Largest share of decisions answerable, most confident first, at error <= max_err."""
    if not pairs:
        return None
    best, wrong = 0, 0
    for k, (_, correct) in enumerate(sorted(pairs, key=lambda x: -x[0]), 1):
        wrong += not correct
        if wrong / k <= max_err:
            best = k
    return best / len(pairs)


def bootstrap_ci(values, reps=2000, seed=0):
    rng, n = random.Random(seed), len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def mcnemar(b, c):
    n = b + c
    return None if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def parse_console_times(path):
    """Phase timestamps from a run_jev.sh console log (for runs made before timing.json)."""
    marks, pats = {}, [("start", r"flow -> "), ("backend_healthy", r"(vLLM|CPU backend) healthy"),
                       ("gateway_healthy", r"gateway healthy"), ("bench_start", r"JevBench: 231"),
                       ("load_start", r"load test:"), ("report_done", r"flow complete")]
    for line in open(path):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)", line)
        if not m:
            continue
        for name, pat in pats:
            if name not in marks and re.search(pat, m.group(2)):
                marks[name] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return marks


def fmt(x, spec=".3f", dash="—"):
    return dash if x is None else format(x, spec)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="'label|conditions|run_dir[|console_log]'; repeat per run")
    ap.add_argument("--repeat", action="append", default=[],
                    help="'label A||label B': two runs of the same configuration, for repeatability")
    ap.add_argument("--jevbench-dir", default="datasets/jevbench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.jevbench_dir))
    from jevbench.metrics import percentile
    from jevbench.tasks import load_jsonl

    tasks = {}
    for tier, f in TIER_FILES:
        for t in load_jsonl(os.path.join(args.jevbench_dir, "datasets/public", f)):
            tasks[t.id] = (tier, t)
    ids_all = list(tasks)
    ids_213 = [i for i in ids_all if tasks[i][1].question["type"] != "score"]
    types = {"choice": "Choice", "noul": "Yes/no", "score": "Ordinal"}

    per_task = json.load(open(os.path.join(args.jevbench_dir, "results/v1.2/jevbench-v1.2-per-task.json")))
    published = {r["key"]: r for r in json.load(open(os.path.join(
        args.jevbench_dir, "results/v1.2/jevbench-v1.2-results.json")))["systems"]}

    systems = []   # dicts: label, kind, correct{id: bool}, conf{id: float}, ...
    for spec in args.run:
        parts = spec.split("|")
        label, cond, run_dir = parts[0], parts[1], parts[2]
        console = parts[3] if len(parts) > 3 else None
        recs = {r["task_id"]: r for r in map(json.loads, open(os.path.join(run_dir, "results.jsonl")))}
        man = json.load(open(os.path.join(run_dir, "manifest.json")))
        rd = lambda name: json.load(open(os.path.join(run_dir, name))) if os.path.exists(os.path.join(run_dir, name)) else None
        load, flip, timing, env = rd("load.json"), rd("orderflip.json"), rd("timing.json"), rd("env.json")
        marks = (timing or {}).get("marks") or (parse_console_times(console) if console else {})
        lat = [recs[i]["latency_s"] for i in ids_all if i in recs]
        bench_wall = (datetime.fromisoformat(man["finished_utc"]) - datetime.fromisoformat(man["started_utc"])).total_seconds()
        serial = next((lv for lv in (load or {}).get("levels", []) if lv["concurrency"] == 1), None)
        peak = max((load or {}).get("levels", []), key=lambda lv: lv["decisions_per_s"] or 0, default=None)
        systems.append({
            "label": label, "kind": "run", "conditions": cond, "run_dir": run_dir,
            "correct": {i: bool(recs.get(i, {}).get("correct")) for i in ids_all},
            "predicted": {i: recs.get(i, {}).get("predicted") for i in ids_all},
            "max_latency": max(lat) if lat else None,
            "n_over_1s": sum(x > 1.0 for x in lat),
            "conf": {i: max(recs[i]["probs"].values()) for i in ids_all if recs.get(i, {}).get("probs")},
            "mass": [m for i in ids_213 for m in ((recs.get(i, {}).get("runtime") or {}).get("class_mass") or {}).values()],
            "p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95),
            "mean_latency": statistics.mean(lat) if lat else None,
            "bench_wall_s": bench_wall,
            "load_to_healthy_s": (marks["backend_healthy"] - marks["start"]) if {"start", "backend_healthy"} <= set(marks) else None,
            "serial_dps": serial["decisions_per_s"] if serial else (len(lat) / bench_wall if bench_wall else None),
            "peak": peak, "flip": flip, "timing": timing, "env": env,
        })
    for key, label in BASELINES:
        row = per_task["systems"].get(key)
        if row is None:
            continue
        pt = row["public_tasks"]
        conf = {i: pt[i][1] for i in ids_all if i in pt and isinstance(pt[i][1], (int, float))}
        if not all(0 <= c <= 1 for c in conf.values()):
            conf = {}   # not a probability for this system (e.g. a raw margin): no calibration
        speed = (published.get(key) or {}).get("speed") or {}
        systems.append({
            "label": label, "kind": "published", "key": key,
            "correct": {i: i in pt and pt[i][0] == "c" for i in ids_all},
            "attempted": {i: i in pt and pt[i][0] != "n" for i in ids_all},
            "conf": conf, "p50": speed.get("p50_s_raw"), "p95": speed.get("p95_s_raw"),
            "endpoint": (published.get(key) or {}).get("endpoint_kind"),
        })

    def acc(s, ids):
        return sum(s["correct"][i] for i in ids) / len(ids) if ids else None

    def pairs(s, ids):
        return [(s["conf"][i], s["correct"][i]) for i in ids if i in s["conf"]]

    runs = [s for s in systems if s["kind"] == "run"]
    pubs = [s for s in systems if s["kind"] == "published"]
    jev = next(s for s in pubs if s["key"] == "jev-1.13.0")
    out = []

    # ---- Table 1: LLM2Jev layout ------------------------------------------------------- #
    out += ["### Table 1 — JevBench public, all 231 items (LLM2Jev layout)", "",
            "| System | Accuracy | Latency P50 (s) | Latency P95 (s) | Measured on |",
            "|---|---:|---:|---:|---|"]
    for s in runs:
        out.append(f"| **{s['label']}** | **{pct(acc(s, ids_all))}** | {fmt(s['p50'])} | {fmt(s['p95'])} | {s['conditions']}; serial, localhost |")
    out.append(f"| {LLM2JEV_OWN[0]} | {LLM2JEV_OWN[1]:.1f}% | {LLM2JEV_OWN[2]:.3f} | {LLM2JEV_OWN[3]:.3f} | local endpoint, hardware not stated |")
    for s in pubs:
        where = {"api": "production API, over the internet", "demo": "public demo, over the internet",
                 "gpu": "JevBench's GPU pod, over the internet", "cpu": "JevBench's CPU, over the internet"}.get(s["endpoint"], "JevBench run")
        lat_note = f"{where} (standard+judge serial run)" if s["p50"] is not None else "—"
        out.append(f"| {s['label']} | {pct(acc(s, ids_all))} | {fmt(s['p50'])} | {fmt(s['p95'])} | {lat_note} |")
    out += ["", "Accuracy of published systems: scored from JevBench's per-item outcomes "
            "(failures count wrong). This reproduces every row of LLM2Jev's table except Jev 1.13.0, "
            f"for which LLM2Jev quotes {LLM2JEV_REPORTED['jev-1.13.0']}% and the outcomes give "
            f"{sum(jev['correct'].values())}/231 = {pct(acc(jev, ids_all))}.", ""]

    # ---- Table 2: AnyJev Table 2 layout ------------------------------------------------ #
    out += ["### Table 2 — JevBench public, 213 items (AnyJev Table 2 layout)", "",
            "| Model | Answer mass | Raw | Debiased | ECE raw | ECE debiased | ECE + temp. |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for s in runs:
        mass = statistics.mean(s["mass"]) if s["mass"] else None
        out.append(f"| **{s['label']}** | {fmt(mass)} | **{fmt(acc(s, ids_213))}** | — | {fmt(ece_equal_mass(pairs(s, ids_213)))} | — | — |")
    for s in pubs:
        if s["key"] in ("jev-1.13.0", "simplejev-qwen3.8-27b", "jqv", "deepseek-flash"):
            out.append(f"| {s['label']} | — | {fmt(acc(s, ids_213))} | — | {fmt(ece_equal_mass(pairs(s, ids_213)))} | — | — |")
    for r in ANYJEV_TABLE2:
        out.append(f"| {r[0]} — AnyJev, as reported | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} |")
    out += ["", "Raw = one forward pass in the canonical option order, no label-prior correction "
            "and no rotations; SALSA has no debiased or temperature-fitted variant (—). ECE: 15 equal-mass bins "
            "of top-label confidence. Published systems' confidences come from JevBench's per-item file; systems whose "
            "per-item value is not a probability have no ECE.", ""]

    # ---- Table 3: per tier --------------------------------------------------------------- #
    tiers_213 = {t: [i for i in ids_213 if tasks[i][0] == t] for t, _ in TIER_FILES}
    tiers_231 = {t: [i for i in ids_all if tasks[i][0] == t] for t, _ in TIER_FILES}
    out += ["### Table 3 — accuracy by tier (AnyJev Table 9 layout; scored items 48 / 60 / 105)", "",
            "| Model | easy (48) | original (72) | hard (111) |", "|---|---:|---:|---:|"]
    for s in runs + [p for p in pubs if p["key"] in ("jev-1.13.0", "simplejev-qwen3.8-27b", "jqv", "deepseek-flash")]:
        b = "**" if s["kind"] == "run" else ""
        out.append(f"| {b}{s['label']}{b} | " + " | ".join(fmt(acc(s, tiers_213[t])) for t, _ in TIER_FILES) + " |")
    for r in ANYJEV_TABLE9:
        out.append(f"| {r[0]}, as reported | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} |")
    out += ["", "Same, JevBench protocol (all 231 items, score items included):", "",
            "| Model | easy (48) | standard (72) | hard (111) | all (231) |", "|---|---:|---:|---:|---:|"]
    for s in runs + [p for p in pubs if p["key"] in ("jev-1.13.0", "simplejev-qwen3.8-27b", "deepseek-flash")]:
        b = "**" if s["kind"] == "run" else ""
        out.append(f"| {b}{s['label']}{b} | " + " | ".join(pct(acc(s, tiers_231[t])) for t, _ in TIER_FILES)
                   + f" | {pct(acc(s, ids_all))} |")
    out.append("")

    # ---- Table 4: per question type ----------------------------------------------------- #
    by_type = {k: [i for i in ids_all if tasks[i][1].question["type"] == k] for k in types}
    out += ["### Table 4 — accuracy by question type (all 231 items)", "",
            "| Model | " + " | ".join(f"{types[k]} ({len(v)})" for k, v in by_type.items()) + " |",
            "|---|" + "---:|" * len(by_type)]
    for s in runs + [p for p in pubs if p["key"] in ("jev-1.13.0", "simplejev-qwen3.8-27b", "deepseek-flash")]:
        b = "**" if s["kind"] == "run" else ""
        out.append(f"| {b}{s['label']}{b} | " + " | ".join(pct(acc(s, v)) for v in by_type.values()) + " |")
    out.append("")

    # ---- Table 5: uncertainty and paired comparisons --------------------------------------- #
    out += ["### Table 5 — 95% bootstrap intervals and paired comparisons (231 items, 2,000 resamples)", "",
            "| Comparison | Accuracy A | 95% CI | Accuracy B | Δ (A − B) | 95% CI of Δ | A-only / B-only | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    comps = [(r, jev) for r in runs]
    simple = next((p for p in pubs if p["key"] == "simplejev-qwen3.8-27b"), None)
    comps += [(r, simple) for r in runs if simple and "Qwen3.8" in r["label"]]
    comps += [(a, b) for k, a in enumerate(runs) for b in runs[k + 1:]]
    for a, b in comps:
        xa = [int(a["correct"][i]) for i in ids_all]
        xb = [int(b["correct"][i]) for i in ids_all]
        lo, hi = bootstrap_ci(xa)
        dlo, dhi = bootstrap_ci([p - q for p, q in zip(xa, xb)])
        ao = sum(p and not q for p, q in zip(xa, xb)); bo = sum(q and not p for p, q in zip(xa, xb))
        out.append(f"| {a['label']} vs {b['label']} | {pct(sum(xa) / 231)} | [{100 * lo:.1f}, {100 * hi:.1f}] | "
                   f"{pct(sum(xb) / 231)} | {100 * (sum(xa) - sum(xb)) / 231:+.1f} pt | [{100 * dlo:+.1f}, {100 * dhi:+.1f}] | "
                   f"{ao} / {bo} | {fmt(mcnemar(ao, bo))} |")
    out.append("")

    # ---- Table 6: calibration, selective prediction, option order ------------------------- #
    out += ["### Table 6 — calibration, selective answering and option-order robustness", "",
            "| Model | ECE (213, 15 eq-mass) | ECE hard (JevBench: 10 bins) | mean confidence (hard) | accuracy (hard) | cov@5% (213) | order-flip rate (choice) | accuracy, options reversed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    from jevbench.metrics import ece_top_label
    hard = tiers_231["hard"]
    for s in runs + [jev]:
        hp = pairs(s, hard)
        f = s.get("flip")
        flip = f"{100 * f['order_flip_rate']:.1f}% ({f['flips']}/{f['n_choice_items']})" if f else "—"
        out.append(f"| {s['label']} | {fmt(ece_equal_mass(pairs(s, ids_213)))} | "
                   f"{fmt(ece_top_label(hp)['ece'] if hp else None)} | {fmt(statistics.mean(p for p, _ in hp) if hp else None)} | "
                   f"{fmt(acc(s, hard))} | {pct(cov_at(pairs(s, ids_213)))} | {flip} | "
                   f"{fmt(f['accuracy_reversed_order']) if f else '—'} |")
    out.append("")

    # ---- Table 7: run time and serving ----------------------------------------------------- #
    out += ["### Table 7 — run time and serving (AnyJev Table 3 style, plus wall-clock phases)", "",
            "| System | Engine / hardware | Model load → healthy (s) | Serial run, 231 decisions (s) | Mean / P50 / P95 / max latency (s) | Decisions > 1 s | Decisions/s serial | Peak decisions/s (concurrency) | Prompt tokens/s at peak |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in runs:
        pk = s["peak"]
        out.append(f"| {s['label']} | {s['conditions']} | {fmt(s['load_to_healthy_s'], '.0f')} | {s['bench_wall_s']:.0f} | "
                   f"{fmt(s['mean_latency'])} / {fmt(s['p50'])} / {fmt(s['p95'])} / {fmt(s['max_latency'], '.2f')} | "
                   f"{s['n_over_1s']} | {fmt(s['serial_dps'], '.2f')} | "
                   f"{fmt(pk['decisions_per_s'], '.2f') if pk else '—'} ({pk['concurrency'] if pk else '—'}) | "
                   f"{fmt(pk['input_tokens_per_s'], ',.0f') if pk else '—'} |")
    out += ["", "AnyJev, as reported (MASSIVE 18-option routing, 300 decisions, one H100, serial): vLLM 0.7.0 reading "
            "all 18 rotations 16.7 decisions/s; with its stopping rule 37.2 decisions/s. LLM2Jev reports latency only.", ""]
    for s in runs:
        if s["timing"]:
            out.append(f"- {s['label']} phases: " + ", ".join(f"{k.replace('_s', '')} {v:.0f} s"
                                                             for k, v in s["timing"]["phases_s"].items())
                       + f"; total {s['timing']['total_s']:.0f} s")
    out.append("")

    # ---- Table 8: repeatability ---------------------------------------------------------- #
    if args.repeat:
        out += ["### Table 8 — repeatability: the same configuration run twice", "",
                "| Runs | Identical answers | Accuracy (first / second) | First-only / second-only correct | McNemar p |",
                "|---|---:|---:|---:|---:|"]
        by_label = {s["label"]: s for s in runs}
        for pair in args.repeat:
            a, b = (by_label[x] for x in pair.split("||"))
            same = sum(a["predicted"][i] == b["predicted"][i] for i in ids_all)
            ao = sum(a["correct"][i] and not b["correct"][i] for i in ids_all)
            bo = sum(b["correct"][i] and not a["correct"][i] for i in ids_all)
            out.append(f"| {a['label']} → {b['label']} | {same}/231 | {pct(acc(a, ids_all))} / {pct(acc(b, ids_all))} | "
                       f"{ao} / {bo} | {fmt(mcnemar(ao, bo))} |")
        out.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(out))
    print("\n".join(out))
    if args.json:
        dump = [{k: v for k, v in s.items() if k not in ("correct", "conf", "attempted", "mass", "predicted")}
                | {"accuracy_231": acc(s, ids_all), "accuracy_213": acc(s, ids_213)} for s in systems]
        json.dump(dump, open(args.json, "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
