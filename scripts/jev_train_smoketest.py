#!/usr/bin/env python3
"""CPU smoke test of the Jev training path: train -> merge -> serve -> same answers.

Builds a tiny, randomly initialised copy of a real architecture (real tokenizer and
chat template), then:

  1. option shuffling: the gold option's letter moves, the target follows the option;
  2. trains a LoRA adapter on examples/jev_decisions_example.jsonl with train_jev.py and
     requires it to fit the data (validation = training data; this proves the gradient
     path, not generalisation);
  3. merges the adapter, serves the merged model through the real CPU backend and the
     /v1/systemone gateway, and requires the served probabilities to equal the
     trainer's own readout -- training and serving use one prompt and one readout.

    python scripts/jev_train_smoketest.py --base Qwen/Qwen3-0.6B      # ~2 min on CPU
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "examples", "jev_decisions_example.jsonl")


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print(f"  ok  {msg}", flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_http(url, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"FAIL: server exited during startup ({url})")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"FAIL: {url} not healthy after {timeout} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="architecture + tokenizer to miniaturise")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--workdir", default=None, help="default: a temporary directory")
    ap.add_argument("--epochs", type=float, default=60)
    args = ap.parse_args(argv)

    import torch
    from make_tiny_model import build
    from src.jev.decision import PromptBuilder, parse_question
    from src.jev.train import DecisionCollator, DecisionDataset, evaluate, load_records

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    work = args.workdir or tempfile.mkdtemp(prefix="jev_train_smoke_")
    tiny_dir, out_dir, merged_dir = (os.path.join(work, d) for d in ("tiny", "adapter", "merged"))

    print(f"[1/4] tiny copy of {args.base} -> {tiny_dir}", flush=True)
    model, tok = build(args.base, args.cache_dir, n_layers=2, seed=0)
    model.save_pretrained(tiny_dir)
    tok.save_pretrained(tiny_dir)
    builder = PromptBuilder(tok)

    print("[2/4] option shuffling", flush=True)
    records, problems = load_records(DATA)
    check(not problems and len(records) == 30, f"example data loads: {len(records)} decisions, {len(problems)} problems")
    ds = DecisionDataset(records, builder, 4096, shuffle_options=True, seed=1)
    rec = next(r for r in records if r["question"]["type"] == "choice")
    letters_seen = set()
    for _ in range(12):
        e = ds.encode(rec, shuffle=True)
        text = tok.decode(e.input_ids)
        gold = e.class_ids[max(range(len(e.target)), key=e.target.__getitem__)]
        letter = tok.decode([gold])
        check_line = f"{letter}. {rec['expected']}"
        if check_line not in text:
            raise SystemExit(f"FAIL: target letter {letter!r} does not point at {rec['expected']!r}")
        letters_seen.add(letter)
    check(len(letters_seen) > 1, f"gold option moves between letters ({sorted(letters_seen)}) and the target follows it")
    canon = ds.encode(rec, shuffle=False)
    check(parse_question(rec["question"]).keys == tuple(rec["question"]["criteria"]), "evaluation keeps the given order")
    del canon

    print("[3/4] train -> merge", flush=True)
    from train_jev import main as train_main

    rc = train_main(["--model", tiny_dir, "--train", DATA, "--val", DATA, "--out", out_dir,
                     "--epochs", str(args.epochs), "--lr", "1e-2", "--batch-size", "8", "--grad-accum", "1",
                     "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.0", "--no-shuffle-options",
                     "--dtype", "float32", "--logging-steps", "20", "--merge-to", merged_dir])
    check(rc == 0, "train_jev.py finished")
    report = json.load(open(os.path.join(out_dir, "train_report.json")))
    before, after = report["val_before"]["all"], report["val_final"]["all"]
    # A random 2-layer model with a ~65k-parameter adapter: this proves that gradients reach the
    # class-token readout, not that the method learns well (that needs real weights).
    check(after["nll"] < before["nll"] * 0.85, f"validation NLL fell {before['nll']:.3f} -> {after['nll']:.3f}")
    check(after["accuracy"] >= before["accuracy"] + 0.15,
          f"accuracy rose {before['accuracy']:.2f} -> {after['accuracy']:.2f}")
    for f in ("best/adapter_model.safetensors", "best/adapter_config.json", "best/salsa_jev.json"):
        check(os.path.exists(os.path.join(out_dir, f)), f"{f} written")

    from transformers import AutoModelForCausalLM

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    ref = evaluate(merged, DecisionDataset(records, builder, 4096, shuffle_options=False),
                   DecisionCollator(tok.pad_token_id or tok.eos_token_id), 8)
    check(abs(ref["all"]["accuracy"] - after["accuracy"]) < 1e-9, "merged model reproduces the adapter's accuracy")

    print("[4/4] serve the merged model: CPU backend + /v1/systemone gateway", flush=True)
    py = sys.executable
    bport, gport = free_port(), free_port()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen([py, os.path.join(ROOT, "scripts", "jev_cpu_backend.py"), "--scorer", "hf",
                               "--model-dir", merged_dir, "--port", str(bport), "--served-model-name", "tiny",
                               "--min-free-gb", "1", "--dtype", "float32"], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    try:
        wait_http(f"http://127.0.0.1:{bport}/health", procs[0])
        procs.append(subprocess.Popen([py, "-m", "src.jev.server", "--backend", f"http://127.0.0.1:{bport}",
                                       "--backend-model", "tiny", "--tokenizer", merged_dir, "--port", str(gport)],
                                      cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wait_http(f"http://127.0.0.1:{gport}/healthz", procs[1])
        worst = 0.0
        for item, rec in zip(ref["items"], records):
            body = {"state": rec["state"], "questions": {"d": rec["question"]}}
            req = urllib.request.Request(f"http://127.0.0.1:{gport}/v1/systemone", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            served = json.loads(urllib.request.urlopen(req, timeout=60).read())["answers"]["d"]["probabilities"]
            worst = max(worst, max(abs(served[k] - item["probs"][k]) for k in served))
        check(worst < 1e-3, f"served probabilities equal the trainer's readout on all 30 decisions (max diff {worst:.1e})")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
    print(f"TRAIN SMOKE TEST PASSED ({work})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
