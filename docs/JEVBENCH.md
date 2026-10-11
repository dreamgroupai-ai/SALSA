# SALSA vs. Jev on JevBench

[JevBench](https://github.com/fstandhartinger/jevbench) measures **Jev-class decision
models**: the caller sends a piece of state and a typed question (`noul` yes/no, `choice`
over named options, ordinal `score`), and the model returns a probability for every
allowed answer. Jev itself is TypeSafe AI's decision model, served behind
`/v1/systemone`; Jev 1.13.0 is #1 on the v1.3 board (score 74.4).

This setup serves **Qwen3.8-27B as a SALSA decision engine** behind the same
`/v1/systemone` wire format, and scores it with the benchmark's own harness and
adapter. One forward pass per decision, with the answer read from the class-token logits.

Nothing in this document launches a GPU job. `./run_jev.sh check` is CPU-only.

**Results:** [JEVBENCH_RESULTS.md](JEVBENCH_RESULTS.md) has the GPU runs (4× B200, 2026-10-08) against Jev, LLM2Jev and AnyJev: accuracy, calibration, order-flip, latency, throughput and run time.

---

## 1. What is compared, on which items

| | items | public? |
|---|---|---|
| easy | 48 + 24 held out | `datasets/public/easy.jsonl` |
| standard | 72 + 24 held out | `datasets/public/original.jsonl` |
| judge | 146 | **not public** (imported task text) |
| hard | 111 + 109 held out | `datasets/public/hard.jsonl` |

The 231 public decisions are what can be run here. The benchmark publishes per-item
outcomes of every system on exactly these items
(`results/v1.2/jevbench-v1.2-per-task.json`), so the comparison with Jev is on
**identical items**, with an exact McNemar test on the discordant ones.

Two reference rows:

* **Jev 1.13.0 (TypeSafe AI)**, the system to beat.
* **SimpleJev Qwen3.8-27B (Featherless)**: the same base model, also read as a
  direct-logit classifier (v1.3 score 66.3, hard tier 75.0%). It is the control: a gap
  to this row isolates prompt and readout, and a gap to Jev is the headline.

The output is a **public-subset proxy**. It uses the benchmark's own v1.3 functions,
but has no judge tier and no held-out items, so it is never an official JevBench Score.

**On MoE.** Qwen3.8-27B is dense: a `qwen3_5` hybrid with 48 Gated-DeltaNet layers
and 16 full-attention layers, and no experts. vLLM's MoE improvements therefore do not
apply to it. The upgrade still matters because the system vLLM (0.16) predates this
architecture's current support. `EP=1` turns on expert parallelism for MoE checkpoints
such as `Qwen/Qwen3-Next-80B-A3B-Instruct`, which is also in `cache_model/`.

---

## 2. Serving

```
jevbench.cli run --adapter typesafe
        │  POST /v1/systemone {"state", "model", "questions": {"decision": {type, instructions, criteria}}}
        ▼
src/jev/server.py  (:8000)   render SALSA prompt -> token ids; check context; softmax over classes
        │  POST /v1/completions {prompt: ids, max_tokens: 1, logprob_token_ids: class ids}
        ▼
vllm serve Qwen3.8-27B (:8100)   vLLM 0.31.0, bf16, text-only
```

The response is what the benchmark's `typesafe` adapter parses:

```json
{"model": "salsa-qwen3.8-27b",
 "answers": {"decision": {"type": "choice", "choice": "refund",
                          "probabilities": {"cancel": 0.01, "refund": 0.93, "...": 0.06}}},
 "usage": {"input_tokens": 177, "output_tokens": 1},
 "runtime": {"prompt_version": "salsa-jev-v1", "class_mass": {"decision": 0.98},
             "timing_ms": {"render": 0.7, "engine": 21.3, "total": 22.4}}}
```

`noul` answers carry `"noul": P(yes)`, and `score` answers carry `"score": level` plus
probabilities keyed `"0"`…`"n-1"`. A request may hold several named questions; they
run concurrently. The harness stores `runtime` for every item, so the server-side
split ships with the results.

### Decisions that are not obvious

**The answer keys come from `criteria`, not from a label list.** The adapter sends only
`type`, `instructions` and `criteria`. Choice keys are the criteria keys, noul is
`yes`/`no`, and score levels are the list indices. The self-check verifies on all 231
items that this reproduces the label set exactly. Criteria arrive in alphabetical key
order (the benchmark stores canonical sorted JSON); that is the order Jev sees too, and
it is the order presented to the model.

**Class tokens and the anchor.** Options are presented as `A.`, `B.`, … (at most 26),
noul as `yes`/`no`, and score levels as their own digits (at most 10). The prompt ends
at `<ANSWER>\n`, not at `<ANSWER>`, because Qwen's pre-tokenizer glues punctuation onto
a following letter: `<ANSWER>A` tokenizes as `[…, '>A']`, so a standalone `A` would
never be the natural next token there. After the newline every class token stands
alone. `PromptBuilder.check_alignment` asserts this on the real tokenizer at server
startup and on every public item in the self-check. Thinking is off: the empty
`<think></think>` block is rendered, as in the repo's other SALSA pipelines.

**The output head runs in fp32.** A bf16 `lm_head` puts every logit on the bf16 grid
(1/16 to 1/4 apart), which turns near-ties into exact 50/50 ties that the scorer then
breaks by label order. vLLM gets `--hf-overrides '{"head_dtype": "float32"}'`; the CPU
backend recomputes the final projection from an fp32 copy of the head.

**The readout uses `logprob_token_ids`** (exposed by the pinned vLLM 0.31 API; the system 0.16 does not have it). It returns the raw,
full-vocabulary logprobs of exactly the class tokens, whatever their rank and before any
sampling transform. The alternative, `allowed_token_ids` plus top-k logprobs, depends on
vLLM's logprobs mode. Under the server defaults it would also apply the model's
`generation_config` (`top_p`/`top_k`), which can silently drop a low-probability
option. As a by-product, `class_mass`, the total probability on valid answers, is
reported per item: a direct measure of whether the model follows the answer format.

**No truncation and no repair.** A prompt that does not fit `MAX_MODEL_LEN` (32,768;
the longest public prompt is 3,979 tokens) gets HTTP 422, which the harness scores as
wrong rather than as an outage. Malformed questions get 400; backend failures get
502/503, which count toward the harness's "three consecutive infrastructure errors →
stop" rule. A missing class logprob is an error, never a zero.

**`TEMPERATURE` (default 1.0)** divides the class logits before the softmax, which is
the cheapest calibration lever. 1.0 is the model's own distribution. Any other value
must be chosen without looking at the items being reported (the benchmark's reranker
rows fitted theirs on public items and disclosed it), and it is recorded in every
response and in `env.json`.

### Environment

vLLM 0.31.0 (2026-10-05) pins torch 2.13 + CUDA 13, transformers 5.10–5.17 and its own
kernels. Rather than fight the SALSA training environment's pins (`pyproject.toml`), vLLM
gets a **fresh, isolated venv**, `.venv-vllm` (~8 GB, git-ignored), created by
`./run_jev.sh setup`. It needs an NVIDIA driver with CUDA 13.0 support (R580+). The
gateway and `train_jev.py` run in the same venv; `setup` adds `accelerate`, `peft` and
`datasets` with `--no-deps`, so the vLLM pins never move.

Some models JIT-compile kernels at start-up (FlashInfer, DeepGEMM).
They need `nvcc` ≥ 12.9, with headers that match the CUDA runtime exactly. `setup`
therefore also builds a matched CUDA 13.0 toolkit for vLLM alone in
`.venv-vllm/cuda13.0-jit`, whatever CUDA the system has.

JevBench is pinned at commit `bb05a33` in `datasets/jevbench`, and the self-check
verifies the public files against the benchmark's own SHA-256 manifest.

---

## 3. What gets measured

| what | how | where |
|---|---|---|
| accuracy per tier / family | `jevbench.cli run`, `typesafe` adapter: serial, no retries, raw evidence kept | `results.jsonl`, `raw/` |
| calibration | ECE (top label, hard tier) + TVD to the exact gold distributions of the 10 public `probability` items, i.e. v1.3 Calibration | `report.json` |
| serial latency | the same run: p50/p95 over the standard tier, raw and with the board's self-host rule (×2 + 0.15 s) | `report.json` |
| throughput under load | closed-loop sweep (`CONCURRENCY`): decisions/s, prompt tokens/s, latency p50/p90/p95/p99/max, vLLM vs gateway split, vLLM queue/prefill/TTFT from `/metrics` | `load.json` |
| cost | estimate: measured tokens × $0.09/M input (the reference the board used for SimpleJev Qwen3.8-27B); optionally GPU time × `GPU_HOUR_USD` / measured throughput | `report.json`, `load.json` |

**Latency is not comparable one-to-one.** Jev's 0.65 s p50 was measured over the internet
from Germany, while ours is measured on localhost. The board's ×2 + 0.15 s self-host
adjustment exists to bridge exactly that gap. Its README calls the adjustment "an
assumption, not a measurement" and lists a load measurement as planned. The sweep's
`p50 / serial` column is that measurement for this server: it shows at what concurrency
the ×2 factor is actually reached.

**The cost axis is an artifact of the reference price.** At about 700 input tokens per
decision on the public mix, $0.09/M gives about $0.063 per 1,000 decisions, which is a
Cost axis just under 50. Jev pays $0.042/M. Read Cost as the board would price this
model, not as a property of SALSA.

---

## 4. Running it

```bash
./run_jev.sh setup        # once: .venv-vllm with vllm==0.31.0 + JevBench @ bb05a33 (network, CPU)
./run_jev.sh check        # CPU, ~1 min: self-check + full bench/load/report path on a mock vLLM
./run_jev.sh gpu-check    # are the GPUs in $GPUS free?  (`serve`/`flow` refuse busy GPUs)

# GPU -- the experiment, one command: serve -> warm-up -> bench -> load -> report -> stop
./run_jev.sh flow
GPUS=0,1 TP=2 ./run_jev.sh flow                   # lower serial latency
GPUS=0,1,2,3,4,5,6,7 DP=8 ./run_jev.sh flow       # throughput across the box
GPU_HOUR_USD=<price> ./run_jev.sh flow            # adds $/1,000 decisions from GPU time

# or interactively: serve in one shell, measure from another
./run_jev.sh serve
./run_jev.sh bench
./run_jev.sh load logs/jev/bench_<time>          # adds load.json to that run and re-reports
```

`check` runs the real gateway, the real JevBench CLI and adapter, the load test and the
report against `scripts/jev_cpu_backend.py --scorer hash`. That stand-in implements
the `/v1/completions` contract the gateway relies on (verified against the vLLM 0.31
source) and rejects anything else. Its "model" is a hash of the prompt, so the
accuracies in a check run are noise by construction.

### On CPU

```bash
BACKEND=cpu ./run_jev.sh flow      # same harness, gateway and report; the backend is swapped
```

`scripts/jev_cpu_backend.py --scorer hf` serves the real checkpoint behind the same
`/v1/completions` contract: bf16 weights, transformers, one forward pass per decision,
and the identical readout. Accuracy and calibration from a CPU run are therefore real
results. Its latency and throughput describe this CPU, not the GPU service: the report
labels the endpoint `cpu`, and its Speed and Cost axes should not be compared with Jev's.

One forward pass saturates the CPU, so concurrency adds queueing, not throughput. The CPU
flow's load sweep is therefore `1,2` with 24 requests per level.

Configuration:

* **An idle socket, one thread per physical core** (`CPU_NODE`, `CPU_SPARE`). On a shared
  machine, pick the quiet socket and leave a few cores spare: one busy core stalls every
  OpenMP region it belongs to.
* **tcmalloc** (`LD_PRELOAD`, system library), when present: glibc unmaps and re-faults
  large temporary buffers after every op.
* **Weights copied into node-local memory** at load: `from_pretrained` leaves them
  mmapped, and they would otherwise page in during the first forwards. First touch by the
  pinned threads places them on the right node, with no `numactl --membind` needed.
* **Every forward on the main thread.** Off the main thread, the triangular solves in
  transformers' Gated-DeltaNet fallback lose their parallelism. HTTP threads only queue
  requests.
* **No `OMP_PROC_BIND`**: binding threads to cores slows the forwards down.

The Gated-DeltaNet layers run transformers' pure-PyTorch fallback (`flash-linear-attention`
is GPU-only).

Every run writes one directory `logs/jev/<label>_<time>/` containing: `vllm.log`,
`vllm_cmd.txt`, `gateway.log`, `env.json` (versions, GPUs, model snapshot, prompt version,
temperature), `bench.log`, `results.jsonl`, `raw/`, `manifest.json`, `load.json`,
`report.json`, `report.md`, and `FLOW_DONE` on success. Servers run in their own
process groups and are torn down on exit, Ctrl-C or failure.

Knobs, all environment variables: `BACKEND` (`vllm` | `cpu`), `CPU_NODE` (1), `CPU_SPARE` (8),
`WARMUP` (16; cpu: 2), `GPUS` (0), `TP` (#GPUS/DP), `DP` (1), `EP` (0),
`MODEL` (`Qwen/Qwen3.8-27B` or a local dir), `MAX_MODEL_LEN` (32768), `GPU_MEM_UTIL` (0.90),
`TEMPERATURE` (1.0), `LORA` (adapter dir, served as module `salsa`; vLLM only), `CONCURRENCY`
(1,2,…,128; cpu: 1,2), `LOAD_REQUESTS` (cpu: 24), `GPU_HOUR_USD`, `VLLM_EXTRA_ARGS`, `PORT` (8000), `VLLM_PORT` (8100),
`GATEWAY_WORKERS` (1; raise for DP runs if the gateway, not vLLM, saturates), `RUN_DIR`, and `FORCE=1`
to start on GPUs another job is using.

---

## 5. Files

```
src/jev/decision.py       typed question -> SALSA prompt, class tokens, readout, typed answer (pure)
src/jev/server.py         /v1/systemone gateway over vLLM's /v1/completions
scripts/jev_selfcheck.py  CPU: dataset hashes, token alignment on all items, readout math,
                          JevBench's own adapter + scorer accept every answer, 400/422 paths
scripts/jev_cpu_backend.py  CPU stand-in for `vllm serve` (contract-enforcing): hash scorer for
                          `check`, the real model in bf16 for BACKEND=cpu
scripts/jev_loadtest.py   closed-loop throughput/latency sweep (stdlib only)
scripts/jev_report.py     proxy axes, head-to-head vs Jev on identical items, load table
scripts/jev_orderflip.py  order-flip rate: every choice item re-asked with its options reversed
scripts/jev_compare.py    cross-run tables in the LLM2Jev / AnyJev layouts, run time, repeatability
src/jev/train.py          decision dataset, option shuffling, single-token loss, evaluation
train_jev.py              train a decision adapter on your data (docs/JEV_TRAINING.md)
scripts/jev_train_smoketest.py  CPU: tiny model, train -> merge -> serve; served == trained readout
scripts/make_tiny_model.py      random miniature of any architecture, for CPU tests
examples/jev_decisions_example.jsonl  30 example decisions in the training format
run_jev.sh                setup / check / check-train / train / serve / bench / load / flow
```

## 6. Verified, and limits

* **GPU runs:** Qwen3.8-27B on 4× B200 (2026-10-08) is in
  [JEVBENCH_RESULTS.md](JEVBENCH_RESULTS.md). Two runs gave the same answers on 231/231
  items.
* **Trained adapters:** `./run_jev.sh check-train` verifies, on a tiny random model, that a
  trained and merged adapter served through the decision API reproduces the trainer's
  readout ([JEV_TRAINING.md §6](JEV_TRAINING.md)). LoRA name mapping is not verified for
  hybrid architectures (Qwen3.5/3.8): serve those as merged models (`--merge-to`).
* **The JevBench numbers are zero-shot SALSA:** what the readout and prompt get from the
  base model, against Jev's trained decision model. An adapter trained on JevBench's public
  items would make the comparison meaningless; `train_jev.py --exclude-overlap` drops them.
* **Option order:** the benchmark's canonical (alphabetical) order, unpermuted.
  Order-flip rates are measured and reported (7.2% for Qwen3.8-27B). Permutation averaging at inference would cost k forward passes per
  decision; `train_jev.py` shuffles options during training instead.
* **No official score:** held-out and judge items cannot be run here, so the proxy can
  differ from the board. JevBench's own README warns that public items can be "trained on
  or selected against".
