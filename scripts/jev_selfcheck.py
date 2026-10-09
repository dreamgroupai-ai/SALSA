#!/usr/bin/env python3
"""CPU self-check for the SALSA JevBench setup.  Exits non-zero on any failure.

  1. the public JevBench files are byte-identical to the benchmark's manifest;
  2. every class token is one token and stays one token after the answer anchor,
     on every public item (the readout depends on it);
  3. every public prompt fits the context window, with no truncation;
  4. the readout math: invariant to how the backend normalised its logprobs,
     refuses a missing class, survives vLLM's -9999 clamp;
  5. every answer shape is accepted by the benchmark's own ``typesafe`` adapter and
     scored valid by its own ``score_task`` -- the adapter's HTTP call is replaced
     by the gateway's answer for that item, nothing else is mocked;
  6. malformed / unsupported questions raise the error that maps to 400 / 422.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.jev.decision import (  # noqa: E402
    BadQuestion, PromptBuilder, UnsupportedQuestion, build_answer, class_distribution, parse_question)

PUBLIC = ("easy", "original", "hard")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--jevbench-dir", default=os.path.join(ROOT, "datasets/jevbench"))
    ap.add_argument("--max-model-len", type=int, default=32768)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.jevbench_dir))
    import jevbench.adapters.typesafe as ts
    from jevbench.scoring import score_task
    from jevbench.tasks import dataset_hash, load_jsonl, sha256_file
    from transformers import AutoTokenizer

    # 1. dataset integrity
    manifest = {s["name"]: s for s in json.load(open(os.path.join(args.jevbench_dir, "datasets/manifest.json")))["splits"]}
    tasks = []
    for name in PUBLIC:
        path = os.path.join(args.jevbench_dir, f"datasets/public/{name}.jsonl")
        split = load_jsonl(path)
        check(sha256_file(path) == manifest[name]["sha256"], f"{name}: file hash differs from manifest")
        check(dataset_hash(split) == manifest[name]["canonical_sha256"], f"{name}: canonical hash differs")
        tasks += [(name, t) for t in split]
    print(f"[1] dataset: {len(tasks)} public decisions match the manifest hashes")

    # 2 + 3. prompts on every public item
    builder = PromptBuilder(AutoTokenizer.from_pretrained(args.tokenizer))
    lengths, encoded = {}, {}
    for name, t in tasks:
        q = parse_question(ts.build_question(t))   # exactly what the adapter sends
        check(set(q.keys) == set(t.labels), f"{t.id}: answer keys {q.keys} != labels {t.labels}")
        builder.check_alignment(t.state, q)
        ids, class_ids = builder.encode(t.state, q)
        check(len(ids) + 1 <= args.max_model_len, f"{t.id}: {len(ids)} tokens exceed {args.max_model_len}")
        lengths.setdefault(name, []).append(len(ids))
        encoded[t.id] = (q, class_ids)
    print("[2] class tokens aligned after the anchor on every item")
    print("[3] prompt tokens: " + ", ".join(
        f"{k} mean {statistics.mean(v):.0f} max {max(v)}" for k, v in lengths.items())
        + f" (context {args.max_model_len})")

    # 4. readout math
    raw = {11: math.log(0.5), 22: math.log(0.2), 33: math.log(0.1)}   # 20% mass elsewhere in the vocab
    restricted = {k: v - math.log(0.8) for k, v in raw.items()}
    a, b = class_distribution(raw, [11, 22, 33]), class_distribution(restricted, [11, 22, 33])
    check(max(abs(x - y) for x, y in zip(a, b)) < 1e-12, "restricted softmax depends on normalisation")
    check(abs(a[0] - 0.625) < 1e-12 and abs(sum(a) - 1) < 1e-12, f"wrong distribution {a}")
    try:
        class_distribution(raw, [11, 22, 44])
        raise AssertionError("missing class token was not refused")
    except KeyError:
        pass
    c = class_distribution({1: -0.01, 2: -9999.0}, [1, 2])
    check(c[1] == 0.0 and abs(sum(c) - 1) < 1e-12, "-9999 clamp mishandled")
    hot, cold = class_distribution(raw, [11, 22, 33], 0.5), class_distribution(raw, [11, 22, 33], 2.0)
    check(hot[0] > a[0] > cold[0], "temperature does not sharpen/flatten")
    print("[4] readout: normalisation-invariant, refuses missing classes, survives the -9999 clamp")

    # 5. the benchmark's own adapter + scorer accept every answer shape
    def fake_post(task):
        q, class_ids = encoded[task.id]
        logits = {cid: -0.3 * i for i, cid in enumerate(class_ids)}
        ans = build_answer(q, class_distribution(logits, class_ids))
        return 200, {"model": "salsa-selfcheck", "answers": {"decision": ans},
                     "usage": {"input_tokens": 1, "output_tokens": 1}}, 0.0

    adapter = ts.TypeSafeAdapter(endpoint="http://selfcheck", key_env="")
    n_valid = 0
    for _, t in tasks:
        ts.http_post_json = lambda url, body, headers, timeout, _t=t: fake_post(_t)
        res = adapter.run(t)
        check(res.ok, f"{t.id}: adapter rejected the answer: {res.error}")
        scored = score_task(res.probs, t)
        check(scored["valid"] and scored["strict_valid"], f"{t.id}: scorer rejected {res.probs}: {scored.get('error')}")
        n_valid += 1
    print(f"[5] jevbench typesafe adapter + score_task: {n_valid}/{len(tasks)} answers ok and strictly valid")

    # 6. error classes
    cases = [
        ({"type": "choice", "instructions": "q", "criteria": {f"o{i}": "d" for i in range(27)}}, UnsupportedQuestion),
        ({"type": "score", "instructions": "q", "criteria": ["d"] * 11}, UnsupportedQuestion),
        ({"type": "choice", "instructions": "", "criteria": {"a": "x", "b": "y"}}, BadQuestion),
        ({"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}}, BadQuestion),
        ({"type": "rank", "instructions": "q"}, BadQuestion),
        ({"type": "choice", "instructions": "q", "criteria": {"only": "x"}}, BadQuestion),
    ]
    for q, exc in cases:
        try:
            parse_question(q)
            raise AssertionError(f"{q} was accepted")
        except exc:
            pass
    print("[6] malformed -> 400, unsupported -> 422")
    print("SELF-CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
