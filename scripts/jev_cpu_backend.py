#!/usr/bin/env python3
"""CPU stand-in for `vllm serve`: the /v1/completions subset the gateway uses.

Implements only the part of vLLM 0.31's OpenAI server that src/jev/server.py
relies on, and rejects any request that leaves that contract:

  POST /v1/completions   prompt = token ids, max_tokens = 1, logprob_token_ids
                         (<= 128 ids, `logprobs` set), return_tokens_as_token_ids
                         -> choices[0].logprobs.top_logprobs[0] =
                            {"token_id:<id>": raw full-vocabulary logprob, ...}
                            for every requested id plus the sampled token,
                            clamped at -9999 like vLLM
  GET  /health, /v1/models, /metrics (the Prometheus families the load test reads)

Two scorers behind that contract:

  --scorer hash       no model; logprobs are a hash of the prompt and the latency is
                      simulated (`--base-ms`, `--us-per-token`, `--slots`).  Answers
                      are noise, the plumbing is real: this is `run_jev.sh check`.
  --scorer hf         the real checkpoint (`--model-dir`) in bf16 on CPU through
                      transformers -- the same weights and the same readout as vLLM
                      on a GPU, one forward pass per decision (`BACKEND=cpu`).
                      Bind it to one NUMA node (run_jev.sh does, with numactl):
                      the GEMMs run on AMX and cross-socket memory traffic only
                      slows them down.  Forward passes are serialised; a CPU is
                      already saturated by one.

Every hf forward runs on the main thread.  Off the main thread the same forward is
many times slower: the triangular solves in transformers' Gated-DeltaNet fallback lose
their parallelism, while the bf16 GEMMs do not.  So HTTP handler threads only queue requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import socket
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.e2e = [0.0, 0]
        self.queue = [0.0, 0]
        self.prompt_tokens = 0

    def add(self, e2e, queue, tokens):
        with self.lock:
            self.e2e[0] += e2e
            self.e2e[1] += 1
            self.queue[0] += queue
            self.queue[1] += 1
            self.prompt_tokens += tokens

    def prometheus(self):
        with self.lock:
            return (f'vllm:e2e_request_latency_seconds_sum{{model_name="cpu"}} {self.e2e[0]}\n'
                    f'vllm:e2e_request_latency_seconds_count{{model_name="cpu"}} {self.e2e[1]}\n'
                    f'vllm:request_queue_time_seconds_sum{{model_name="cpu"}} {self.queue[0]}\n'
                    f'vllm:request_queue_time_seconds_count{{model_name="cpu"}} {self.queue[1]}\n'
                    f'vllm:prompt_tokens_total{{model_name="cpu"}} {self.prompt_tokens}\n')


def contract_error(body, served, max_model_len, vocab):
    if body.get("model") != served:
        return f"model {body.get('model')!r} is not served (serving {served!r})"
    p = body.get("prompt")
    if not (isinstance(p, list) and p and all(isinstance(i, int) and 0 <= i < vocab for i in p)):
        return "prompt must be a non-empty list of in-vocabulary token ids"
    if len(p) + body.get("max_tokens", 16) > max_model_len:
        return f"prompt ({len(p)} tokens) + max_tokens exceeds max_model_len {max_model_len}"
    if body.get("max_tokens") != 1:
        return "max_tokens must be 1"
    ids = body.get("logprob_token_ids")
    if not (isinstance(ids, list) and 0 < len(ids) <= 128 and all(isinstance(i, int) and 0 <= i < vocab for i in ids)):
        return "logprob_token_ids must list 1..128 in-vocabulary ids"
    if body.get("logprobs") is None:
        return "when using `logprob_token_ids`, `logprobs` must be set."
    if body.get("return_tokens_as_token_ids") is not True:
        return "the gateway must ask for return_tokens_as_token_ids"
    return None


class HashScorer:
    """No model: deterministic noise, with simulated latency and batch slots."""

    vocab = 248320  # Qwen3.8; ids outside it are rejected, as in vLLM

    def __init__(self, args):
        self.args = args
        self.slots = threading.BoundedSemaphore(args.slots)

    def __call__(self, prompt, ids):
        with self.slots:
            t_start = time.perf_counter()
            time.sleep((self.args.base_ms + self.args.us_per_token * len(prompt) / 1000) / 1000)
        seed = int.from_bytes(hashlib.sha256(json.dumps(prompt[-256:]).encode()).digest()[:8], "big")
        rng = random.Random(seed)
        logits = [rng.gauss(0.0, 2.0) for _ in ids]
        mass = rng.uniform(0.7, 0.99)   # classes get 70-99% of the vocabulary's mass
        m = max(logits)
        z = m + math.log(sum(math.exp(v - m) for v in logits))
        return [v - z + math.log(mass) for v in logits], t_start


GB = 1e9
# bf16 bytes per stored element once dequantized: packed FP4 (I8) holds two values;
# block scales (E8M0) are folded into the weights and dropped.
BF16_BYTES = {"I8": 4, "F8_E4M3": 2, "BF16": 2, "F16": 2, "F32": 4, "F8_E8M0": 0, "I64": 8, "I32": 4}


def memory_status():
    """(host MemAvailable, container anonymous memory, container limit) in bytes.

    There is no swap on this box, so anonymous memory is what cannot be reclaimed;
    page cache from reading the checkpoint is reclaimable and is not counted.
    """
    avail = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable:"))
    try:
        anon = next(int(l.split()[1]) for l in open("/sys/fs/cgroup/memory.stat") if l.startswith("anon "))
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        lim = None if lim == "max" else int(lim)
    except (OSError, StopIteration):
        anon, lim = None, None
    return avail, anon, lim


def estimate_bf16_bytes(model_dir, num_layers=None):
    """Resident size of the model in bf16, from the safetensors headers alone."""
    import re
    import struct

    index = os.path.join(model_dir, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else \
        [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    total = 0
    for shard in shards:
        with open(os.path.join(model_dir, shard), "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, t in header.items():
            if name == "__metadata__" or name.startswith("mtp."):
                continue
            m = re.search(r"layers\.(\d+)\.", name)
            if num_layers is not None and m and int(m.group(1)) >= num_layers:
                continue
            n = 1
            for d in t["shape"]:
                n *= d
            total += n * BF16_BYTES.get(t["dtype"], 2)
    return total


def start_memory_guard(min_free_gb, cgroup_margin_gb, interval_s=0.5):
    """Exit this process at once if memory would run out for others or for the container.

    os._exit, not an exception: the point is to hand the memory back immediately,
    even while transformers is in the middle of a load.
    """
    def watch():
        while True:
            avail, anon, lim = memory_status()
            if avail < min_free_gb * GB or (lim and anon > lim - cgroup_margin_gb * GB):
                print(f"[cpu-backend] MEMORY GUARD: host available {avail / GB:.0f} GB "
                      f"(floor {min_free_gb}), container anon {anon / GB if anon else 0:.0f} GB of "
                      f"{lim / GB if lim else 0:.0f} -- exiting to protect the host", flush=True)
                os._exit(75)
            time.sleep(interval_s)

    threading.Thread(target=watch, daemon=True, name="memory-guard").start()


def pad_single_row_experts(model) -> bool:
    """Run one-row expert matmuls as two rows, for MoE models.

    transformers' eager MoE loop calls F.linear on each expert's tokens.  An expert that
    receives exactly one token hits oneDNN's M=1 path, which is far slower per expert than
    two or more rows, and short prompts route many experts a single token.  The modeling module's `F` is swapped for a proxy,
    so nothing outside that model file changes.
    """
    import sys

    import torch

    mod = sys.modules[type(model).__module__]
    if not any(name.endswith("Experts") for name in vars(mod)) or not hasattr(mod, "F"):
        return False
    F = torch.nn.functional

    class PadSingleRow:
        def __getattr__(self, name):
            return getattr(F, name)

        @staticmethod
        def linear(x, w, b=None):
            if x.dim() == 2 and x.shape[0] == 1:
                return F.linear(torch.cat([x, x]), w, b)[:1]
            return F.linear(x, w, b)

    mod.F = PadSingleRow()
    return True


def fp32_output_head(model) -> None:
    """Compute the final projection (hidden state -> vocabulary logits) in fp32.

    In bf16 the logits land on the bf16 grid: 1/16 to 1/4 apart at typical magnitudes.
    Near-ties then collapse into exact 50/50 ties that the scorer breaks by label order.  The rest of the model stays bf16; only this one
    matmul, a few ms for the single position read, is redone from an fp32 copy of
    the head.
    """
    import torch

    head = model.get_output_embeddings()
    w32 = head.weight.detach().float()

    def hook(module, args, output):
        return torch.nn.functional.linear(args[0].float(), w32)

    head.register_forward_hook(hook)


class HFScorer:
    """The real checkpoint on CPU: one bf16 forward pass, logits at the last position only."""

    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        torch.set_num_threads(args.threads or len(os.sched_getaffinity(0)))
        config = AutoConfig.from_pretrained(args.model_dir)
        if getattr(args, "truncate_layers", None):
            config.num_hidden_layers = args.truncate_layers      # smoke tests only
        need = estimate_bf16_bytes(args.model_dir, getattr(args, "truncate_layers", None))
        avail, _, _ = memory_status()
        min_free = getattr(args, "min_free_gb", 128)
        if avail - need < min_free * GB:
            raise SystemExit(f"[cpu-backend] refusing to load: needs ~{need / GB:.0f} GB, host has "
                             f"{avail / GB:.0f} GB available, and {min_free} GB must stay free")
        print(f"[cpu-backend] loading ~{need / GB:.0f} GB (bf16) with {avail / GB:.0f} GB available", flush=True)
        start_memory_guard(min_free, getattr(args, "cgroup_margin_gb", 32))
        t0 = time.perf_counter()
        # AutoModelForCausalLM resolves VL checkpoints to their text-only CausalLM class (no
        # vision tower).  FP8/FP4 checkpoints have no CPU kernels anywhere: without a GPU,
        # transformers' FP8 quantizer dequantizes them to bf16 during the load.
        quantized = getattr(config, "quantization_config", None) is not None
        dtype = getattr(torch, getattr(args, "dtype", "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(args.model_dir, config=config, dtype=dtype).eval()
        if not quantized:
            # from_pretrained leaves unquantized weights mmapped from the page cache, which may
            # sit on the other socket and pages in during the first forwards.  Private copies
            # are first-touched by this (pinned) thread, so they land in node-local memory.
            # Dequantized weights are already fresh allocations made here.
            with torch.no_grad():
                for p in self.model.parameters():
                    p.data = p.data.clone()
        self.vocab = self.model.config.vocab_size
        fp32_output_head(self.model)
        if pad_single_row_experts(self.model):
            print("[cpu-backend] MoE: one-row expert matmuls run as two rows", flush=True)
        if "tcmalloc" in os.environ.get("LD_PRELOAD", ""):
            # The load leaves its freed temporaries cached in tcmalloc; hand them back
            # to the host.
            import ctypes

            ctypes.CDLL("libtcmalloc_minimal.so.4").MallocExtension_ReleaseFreeMemory()
        self.jobs = queue.Queue()
        print(f"[cpu-backend] loaded {type(self.model).__name__} in {time.perf_counter() - t0:.0f} s, "
              f"{torch.get_num_threads()} threads on CPUs {sorted(os.sched_getaffinity(0))[:4]}...", flush=True)

    def __call__(self, prompt, ids):
        """Called from HTTP handler threads: queue the forward for the main thread."""
        fut = Future()
        self.jobs.put((prompt, ids, fut))
        return fut.result()

    def forward(self, prompt, ids):
        torch = self.torch
        with torch.inference_mode():
            t_start = time.perf_counter()
            out = self.model(input_ids=torch.tensor([prompt]), logits_to_keep=1, use_cache=False)
            lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            return [float(lp[i]) for i in ids], t_start

    def run_forever(self):
        """The main thread's loop: one forward at a time, in arrival order."""
        while True:
            prompt, ids, fut = self.jobs.get()
            try:
                fut.set_result(self.forward(prompt, ids))
            except Exception as e:  # reported to the waiting request, never kills the loop
                fut.set_exception(e)


def make_handler(args, stats, scorer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            # http.server writes headers and body separately; without this, Nagle +
            # delayed ACK add ~40 ms per keep-alive response and swamp the timings.
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *a):
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"", "text/plain")
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": args.served_model_name}]})
            elif self.path == "/metrics":
                self._send(200, stats.prometheus().encode(), "text/plain")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            t_arrive = time.perf_counter()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path != "/v1/completions":
                return self._send(404, {"error": "not found"})
            err = contract_error(body, args.served_model_name, args.max_model_len, scorer.vocab)
            if err:
                return self._send(400, {"error": {"message": err, "type": "BadRequestError"}})
            prompt, ids = body["prompt"], body["logprob_token_ids"]
            try:
                lps, t_start = scorer(prompt, ids)
            except Exception as e:
                return self._send(500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "InternalServerError"}})
            lps = [max(v, -9999.0) for v in lps]
            sampled = ids[max(range(len(ids)), key=lps.__getitem__)]
            top = {f"token_id:{i}": lp for i, lp in zip(ids, lps)}
            self._send(200, {
                "id": "cmpl-cpu", "object": "text_completion", "model": args.served_model_name,
                "choices": [{"index": 0, "text": f"token_id:{sampled}", "finish_reason": "length",
                             "logprobs": {"tokens": [f"token_id:{sampled}"],
                                          "token_logprobs": [top[f"token_id:{sampled}"]],
                                          "top_logprobs": [top], "text_offset": [0]}}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1,
                          "total_tokens": len(prompt) + 1},
            })
            stats.add(time.perf_counter() - t_arrive, t_start - t_arrive, len(prompt))

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--served-model-name", default="qwen3.8-27b")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--scorer", choices=("hash", "hf"), default="hash")
    ap.add_argument("--model-dir", help="checkpoint for --scorer hf")
    ap.add_argument("--threads", type=int, default=0, help="hf: torch threads (0 = one per CPU in the affinity mask)")
    ap.add_argument("--min-free-gb", type=float, default=128,
                    help="hf: host memory that must stay available; refuse to load / exit below it")
    ap.add_argument("--cgroup-margin-gb", type=float, default=32,
                    help="hf: exit when container anonymous memory comes this close to its limit")
    ap.add_argument("--truncate-layers", type=int, default=None, help="hf, smoke tests only: first N layers")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="hf: weight dtype (float32 for exact comparisons in tests)")
    ap.add_argument("--base-ms", type=float, default=15.0, help="hash: fixed cost of one forward pass")
    ap.add_argument("--us-per-token", type=float, default=5.0, help="hash: prefill cost per prompt token")
    ap.add_argument("--slots", type=int, default=8, help="hash: forward passes that may run concurrently")
    args = ap.parse_args(argv)
    if args.scorer == "hf" and not args.model_dir:
        ap.error("--scorer hf needs --model-dir")
    scorer = HFScorer(args) if args.scorer == "hf" else HashScorer(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args, Stats(), scorer))
    server.daemon_threads = True
    print(f"[cpu-backend] {args.scorer} scorer serving {args.served_model_name} on {args.host}:{args.port}", flush=True)
    if isinstance(scorer, HFScorer):
        threading.Thread(target=server.serve_forever, daemon=True).start()
        scorer.run_forever()
    else:
        server.serve_forever()


if __name__ == "__main__":
    main()
