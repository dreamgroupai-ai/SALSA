#!/usr/bin/env python3
"""Option-order robustness of a /v1/systemone endpoint: the order-flip rate.

Re-asks every public JevBench `choice` question with its options in reverse order
and compares the answer with the one the same server gave in the run's serial pass
(`<run>/results.jsonl`).  The order-flip rate is the share of items whose answer
changes -- the metric AnyJev (arXiv 2610.00831) reports, and the effect JevBench
documents for small models (issue #40).

Only `choice` items: the SALSA prompt always lists yes before no, and score levels
have a fixed meaning by position, so neither can be reordered.

    python scripts/jev_orderflip.py --endpoint http://127.0.0.1:8000 --run-dir logs/jev/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

PUBLIC = ("easy", "original", "hard")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--run-dir", required=True, help="run whose results.jsonl holds the original-order answers")
    ap.add_argument("--jevbench-dir", default="datasets/jevbench")
    ap.add_argument("--model", default="salsa")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.jevbench_dir))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.tasks import load_jsonl

    orig = {r["task_id"]: r for r in map(json.loads, open(os.path.join(args.run_dir, "results.jsonl")))}
    adapter = TypeSafeAdapter(endpoint=args.endpoint, model=args.model)
    url = args.endpoint.rstrip("/") + "/v1/systemone"
    items = []
    for name in PUBLIC:
        for t in load_jsonl(os.path.join(args.jevbench_dir, f"datasets/public/{name}.jsonl")):
            if t.question["type"] != "choice" or t.id not in orig or not orig[t.id].get("valid"):
                continue
            body = adapter.build_request(t)
            q = body["questions"]["decision"]
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                probs = json.loads(resp.read())["answers"]["decision"]["probabilities"]
            rev = max(probs, key=probs.get)
            items.append({"task_id": t.id, "tier": name, "original": orig[t.id]["predicted"], "reversed": rev,
                          "gold": t.expected, "n_options": len(t.labels)})
    n = len(items)
    flips = sum(i["original"] != i["reversed"] for i in items)
    out = {
        "n_choice_items": n,
        "order_flip_rate": flips / n if n else None,
        "flips": flips,
        "accuracy_original_order": sum(i["original"] == i["gold"] for i in items) / n if n else None,
        "accuracy_reversed_order": sum(i["reversed"] == i["gold"] for i in items) / n if n else None,
        "by_tier": {tier: {"n": len(v), "flips": sum(i["original"] != i["reversed"] for i in v)}
                    for tier in PUBLIC if (v := [i for i in items if i["tier"] == tier])},
        "items": items,
    }
    with open(os.path.join(args.run_dir, "orderflip.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[orderflip] {flips}/{n} choice answers change when the options are reversed "
          f"({out['order_flip_rate']:.1%}); accuracy {out['accuracy_original_order']:.3f} -> "
          f"{out['accuracy_reversed_order']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
