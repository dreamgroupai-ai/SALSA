#!/usr/bin/env bash
# SALSA as a Jev-style typed decision engine: any open LLM served by vLLM (or on CPU) behind
# TypeSafe's /v1/systemone API, trained on your decisions, and scored on JevBench.
#
#   ./run_jev.sh setup        network: isolated vLLM venv (pinned) + JevBench checkout (pinned)
#   ./run_jev.sh check        CPU: self-check, then the whole bench/load/report path against a
#                             mock backend -- proves the wiring end to end without a GPU
#   ./run_jev.sh check-train  CPU: train -> merge -> serve on a tiny model; served == trained readout
#   ./run_jev.sh train ARGS   train a decision adapter on your data (train_jev.py; docs/JEV_TRAINING.md)
#   ./run_jev.sh gpu-check    are the GPUs in $GPUS free right now?
#   ./run_jev.sh serve        vLLM + decision gateway in the foreground (Ctrl-C stops both)
#   ./run_jev.sh bench        against a running `serve`: serial JevBench run + report
#   ./run_jev.sh load [RUN]   against a running `serve`: throughput/latency sweep (+ report if RUN)
#   ./run_jev.sh flow         serve -> warm-up -> bench -> order-flip -> load -> report -> stop
#
#   BACKEND=cpu ./run_jev.sh flow|serve   the same with transformers on one CPU socket instead of vLLM
#
# Knobs (environment): MODEL (Hugging Face id or local dir; default Qwen/Qwen3.8-27B), LORA (adapter
# dir from train_jev.py, served as module `salsa`), BACKEND (vllm | cpu), GPUS (0), TP (#GPUS/DP),
# DP (1), EP (0; 1 = expert parallel for MoE), MAX_MODEL_LEN (32768), GPU_MEM_UTIL (0.90),
# TEMPERATURE (1.0), CONCURRENCY (1,2,4,...,128; cpu: 1,2), LOAD_REQUESTS, WARMUP, CPU_NODE (0),
# CPU_SPARE (0), GPU_HOUR_USD, VLLM_EXTRA_ARGS, GATEWAY_WORKERS (1), COMPARE, PORT (8000),
# VLLM_PORT (8100), RUN_DIR, FORCE=1 (start on GPUs another process is using).
set -euo pipefail
# Job control off, always: background jobs then stay in this script's process group,
# so `setsid` never forks and $! is the new group's leader (see start_bg).
set +m

cd "$(dirname "$0")"

VENV="${VENV:-.venv-vllm}"
VLLM_VERSION="0.31.0"
JEVBENCH_DIR="${JEVBENCH_DIR:-datasets/jevbench}"
JEVBENCH_REPO="https://github.com/fstandhartinger/jevbench"
JEVBENCH_REV="${JEVBENCH_REV:-bb05a335bc809e61b20c0f745d25499a82b326fc}"   # 2026-09-29, v1.4.2.2 board

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
MODEL_NAME="${MODEL_NAME:-salsa-${SERVED_NAME}}"
GPUS="${GPUS:-0}"
NGPU="$(awk -F, '{print NF}' <<<"$GPUS")"
DP="${DP:-1}"
TP="${TP:-$((NGPU / DP))}"
EP="${EP:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BACKEND="${BACKEND:-vllm}"
case "$BACKEND" in
  vllm) CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64,128}"; WARMUP="${WARMUP:-16}"
        ENDPOINT_KIND="${ENDPOINT_KIND:-gpu}" ;;
  # One forward pass saturates a socket, so concurrency only measures queueing; a few
  # dozen requests per level are enough to show it.
  cpu)  CONCURRENCY="${CONCURRENCY:-1,2}"; WARMUP="${WARMUP:-2}"; LOAD_REQUESTS="${LOAD_REQUESTS:-24}"
        ENDPOINT_KIND="${ENDPOINT_KIND:-cpu}" ;;
  *)    echo "BACKEND must be vllm or cpu" >&2; exit 2 ;;
esac
CPU_NODE="${CPU_NODE:-0}"      # pick an idle socket on a shared machine
CPU_SPARE="${CPU_SPARE:-0}"    # cores of that node to leave to other processes
PORT="${PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8100}"
read -r -a VLLM_EXTRA <<< "${VLLM_EXTRA_ARGS:-}"

PY="$VENV/bin/python"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/cache_model}"   # the repo's model cache (configs/model)
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date '+%F %T')] $*"; }

free_port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'; }

# The three public splits, in the benchmark's tier order (easy, standard, hard).
tasks_spec() {
  local d="$PWD/$JEVBENCH_DIR/datasets/public"
  echo "$d/easy.jsonl,$d/original.jsonl,$d/hard.jsonl"
}

# A local directory, a cached snapshot, or else the Hugging Face id itself (downloaded on first use).
resolve_model() {
  if [ -d "$MODEL" ]; then echo "$MODEL"; return; fi
  local repo="$HF_HUB_CACHE/models--${MODEL//\//--}"
  if [ -f "$repo/refs/main" ]; then echo "$repo/snapshots/$(cat "$repo/refs/main")"; else echo "$MODEL"; fi
}

has_vllm() {
  [ -x "$PY" ] && "$PY" -c "import importlib.metadata as m, sys; sys.exit(m.version('vllm') != '$VLLM_VERSION')" 2>/dev/null
}

need_venv() {
  has_vllm || { echo "missing $VENV with vllm==$VLLM_VERSION -- run ./run_jev.sh setup" >&2; exit 2; }
  [ -f "$JEVBENCH_DIR/jevbench/cli.py" ] || { echo "missing $JEVBENCH_DIR -- run ./run_jev.sh setup" >&2; exit 2; }
}

setup() {
  if has_vllm; then
    log "$VENV already has vllm $VLLM_VERSION"
  else
    # A fresh venv, not --system-site-packages: vLLM $VLLM_VERSION pins torch 2.13 (CUDA 13),
    # the system stack is torch 2.9.1 + vLLM 0.16 + transformers 4.57 for the repo's older
    # pipelines, and neither may change the other.  The NGC extra index in /etc/pip.conf
    # never resolves here, so it is replaced for this install.
    log "creating $VENV with vllm==$VLLM_VERSION"
    python3 -m venv "$VENV"
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_TRUSTED_HOST= "$VENV/bin/pip" install -q --upgrade pip
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_TRUSTED_HOST= "$VENV/bin/pip" install "vllm==$VLLM_VERSION"
  fi
  # A matched CUDA 13.0 toolkit for vLLM's JIT kernels (see start_vllm), outside the venv's packages.
  [ -x "$VENV/cuda13.0-jit/nvidia/cu13/bin/nvcc" ] || PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_TRUSTED_HOST= \
    "$VENV/bin/pip" install -q --no-deps --target "$VENV/cuda13.0-jit" "nvidia-cuda-nvcc==13.0.88" \
    "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-cccl==13.0.85"
  build_jit_toolkit
  # transformers needs accelerate to load FP8/FP4 checkpoints (BACKEND=cpu); --no-deps keeps
  # pip away from the torch/vLLM pins.
  "$PY" -c "import accelerate" 2>/dev/null || PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_TRUSTED_HOST= \
    "$VENV/bin/pip" install -q --no-deps "accelerate==1.15.0"
  # peft for train_jev.py, so one environment serves and trains.  datasets too: transformers'
  # Trainer probes it, and without the real package `import datasets` would resolve to this
  # repo's datasets/ directory (where JevBench is checked out) and fail.
  "$PY" -c "import peft" 2>/dev/null || PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_TRUSTED_HOST= \
    "$VENV/bin/pip" install -q --no-deps "peft==0.21.2"
  "$PY" -c "import datasets; datasets.Dataset" 2>/dev/null || PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_TRUSTED_HOST= "$VENV/bin/pip" install -q "datasets>=3.2.0"
  if [ ! -d "$JEVBENCH_DIR/.git" ]; then
    log "cloning $JEVBENCH_REPO @ ${JEVBENCH_REV:0:12}"
    git clone -q "$JEVBENCH_REPO" "$JEVBENCH_DIR"
  fi
  git -c safe.directory='*' -C "$JEVBENCH_DIR" checkout -q "$JEVBENCH_REV"
  log "JevBench at $(git -c safe.directory='*' -C "$JEVBENCH_DIR" rev-parse --short HEAD)"
}

# Complete the CUDA 13.0 JIT toolkit into the layout an installed toolkit has.  The compiler
# side (nvcc, crt, nvvm, runtime and CCCL headers) is 13.0 and must stay matched; every other
# header and library (cuBLAS/cuBLASLt, cuRAND, cuFFT, ...) is linked from the venv, i.e. the
# very libraries torch loads, so JIT kernels compile and link against what runs.  Linkers
# want unversioned .so names and lib64/; libcuda.so for -lcuda is the installed driver.
build_jit_toolkit() {
  "$PY" - "$VENV/cuda13.0-jit/nvidia/cu13" <<'PY'
import glob, os, re, sys, sysconfig
jit = os.path.abspath(sys.argv[1])
venv = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
for sub in ("include", "lib"):
    for name in os.listdir(os.path.join(venv, sub)):
        dst = os.path.join(jit, sub, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(venv, sub, name), dst)
for path in glob.glob(os.path.join(jit, "lib", "lib*.so.*")):
    plain = os.path.join(jit, "lib", re.sub(r"\.so\..*$", ".so", os.path.basename(path)))
    if not os.path.lexists(plain):
        os.symlink(os.path.basename(path), plain)
os.makedirs(os.path.join(jit, "lib", "stubs"), exist_ok=True)
stub = os.path.join(jit, "lib", "stubs", "libcuda.so")
if not os.path.lexists(stub):
    os.symlink("/usr/lib/x86_64-linux-gnu/libcuda.so.1", stub)
if not os.path.lexists(os.path.join(jit, "lib64")):
    os.symlink("lib", os.path.join(jit, "lib64"))
print(f"[setup] CUDA JIT toolkit ready: {jit}")
PY
}

# Refuse to start on GPUs someone else is using.  Memory is the signal: the box is
# shared, the other jobs run in another container, and their utilisation flickers.
gpu_check() {
  local busy=0 idx used
  for idx in ${GPUS//,/ }; do
    used="$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "$used" -gt "${GPU_BUSY_MB:-2048}" ]; then
      log "GPU $idx: $used MiB already in use"
      busy=1
    else
      log "GPU $idx: free ($used MiB in use)"
    fi
  done
  if [ "$busy" = 1 ] && [ "${FORCE:-0}" != 1 ]; then
    log "refusing to start on busy GPUs (FORCE=1 overrides)"
    return 5
  fi
}

# --------------------------------------------------------------------------- #
# Process management: every server runs in its own process group, and the group
# is torn down on exit, Ctrl-C or failure -- vLLM spawns engine-core workers that
# would otherwise outlive this script and keep holding GPU memory.
# --------------------------------------------------------------------------- #
PGIDS=()
start_bg() {  # start_bg <logfile> <cmd...>; sets BG_PID, which is also the process group id
  local logfile="$1"; shift
  setsid "$@" > "$logfile" 2>&1 &
  BG_PID=$!
  PGIDS+=("$BG_PID")
}

cleanup() {
  local pg alive i
  [ "${#PGIDS[@]}" -gt 0 ] || return 0
  for pg in "${PGIDS[@]}"; do kill -TERM -- "-$pg" 2>/dev/null || true; done
  for i in $(seq 1 60); do
    alive=0
    for pg in "${PGIDS[@]}"; do
      if kill -0 -- "-$pg" 2>/dev/null; then alive=1; fi
    done
    [ "$alive" = 0 ] && break
    sleep 0.5
  done
  for pg in "${PGIDS[@]}"; do kill -KILL -- "-$pg" 2>/dev/null || true; done
  PGIDS=()
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_http() {  # wait_http <url> <timeout_s> <pid> <name>
  local url="$1" deadline=$((SECONDS + $2)) pid="$3" name="$4"
  until curl -sf -o /dev/null "$url"; do
    kill -0 "$pid" 2>/dev/null || { log "$name exited during startup"; return 3; }
    [ "$SECONDS" -lt "$deadline" ] || { log "$name not healthy after $2 s"; return 3; }
    sleep 2
  done
}

start_vllm() {  # start_vllm <run_dir>
  local model_dir args=()
  model_dir="$(resolve_model)"
  args=(serve "$model_dir"
        --served-model-name "$SERVED_NAME"
        --host 127.0.0.1 --port "$VLLM_PORT"
        --dtype bfloat16
        --tensor-parallel-size "$TP" --data-parallel-size "$DP"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --language-model-only                        # text-only: no vision tower in memory
        # fp32 logits: a bf16 lm_head puts every logit on the bf16 grid (1/16-1/4 apart) and
        # turns near-ties into exact 50/50 ties.
        --hf-overrides '{"head_dtype": "float32"}')
  [ "$EP" = 1 ] && args+=(--enable-expert-parallel)
  BACKEND_MODEL="$SERVED_NAME"
  if [ -n "${LORA:-}" ]; then
    local rank
    rank="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1] + "/adapter_config.json"))["r"])' "$LORA")"
    args+=(--enable-lora --lora-modules "salsa=$LORA" --max-lora-rank "$rank")
    BACKEND_MODEL="salsa"
  fi
  args+=("${VLLM_EXTRA[@]}")
  log "vLLM $VLLM_VERSION: $MODEL on GPUs $GPUS (TP=$TP DP=$DP EP=$EP) port $VLLM_PORT${LORA:+, LoRA $LORA}"
  echo "$VENV/bin/vllm ${args[*]}" > "$1/vllm_cmd.txt"
  # JIT kernels (DeepGEMM, FlashInfer) need nvcc >= 12.9, and CCCL requires the
  # compiler to match the CUDA runtime headers exactly.  The system toolkit is 12.8; the venv mixes
  # nvcc 13.4 (pulled by an extra) with 13.0 runtime headers.  `setup` installs a matched 13.0
  # toolkit -- the CUDA version of torch's cu130 build and of the driver -- used for vLLM only.
  local cuda_home="$PWD/$VENV/cuda13.0-jit/nvidia/cu13"
  [ -x "$cuda_home/bin/nvcc" ] || { log "missing $cuda_home -- run ./run_jev.sh setup"; return 2; }
  CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPUS" \
    start_bg "$1/vllm.log" "$VENV/bin/vllm" "${args[@]}"
  BACKEND_PID=$BG_PID
  wait_http "http://127.0.0.1:$VLLM_PORT/health" "${VLLM_START_TIMEOUT:-1800}" "$BACKEND_PID" vLLM
  log "vLLM healthy"
}

# Physical cores of NUMA node $CPU_NODE (one hyperthread each), minus the first $CPU_SPARE.
# One OpenMP thread per physical core of an idle socket works best; on a shared machine leave
# some cores to other processes -- a busy core stalls every parallel region it is part of.
node_cores() {
  "$PY" - "$CPU_NODE" "$CPU_SPARE" <<'PY'
import sys
node, spare = sys.argv[1], int(sys.argv[2])
def expand(text):
    out = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
cpus = expand(open(f"/sys/devices/system/node/node{node}/cpulist").read())
cores = [c for c in cpus
         if expand(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())[0] == c]
print(",".join(map(str, cores[spare:])))
PY
}

# The real checkpoint on CPU behind the same /v1/completions contract as vLLM
# (scripts/jev_cpu_backend.py).  tcmalloc, when present, keeps the large temporary
# buffers that glibc unmaps and re-faults after every op.  numactl cannot set a memory policy in this container; the
# backend copies the weights into node-local memory itself.
start_cpu() {  # start_cpu <run_dir>
  local cores tcmalloc=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
  [ -z "${LORA:-}" ] || { log "LORA is not supported by the CPU backend"; return 2; }
  cores="$(node_cores)"
  [ -f "$tcmalloc" ] || tcmalloc=""
  log "CPU backend: $MODEL in bf16 on NUMA node $CPU_NODE, $(awk -F, '{print NF}' <<<"$cores") cores" \
      "(${cores%%,*}..${cores##*,})${tcmalloc:+, tcmalloc}, port $VLLM_PORT"
  echo "$cores" > "$1/cpu_cores.txt"
  CUDA_VISIBLE_DEVICES="" LD_PRELOAD="$tcmalloc" start_bg "$1/vllm.log" \
    numactl --physcpubind="$cores" "$PY" scripts/jev_cpu_backend.py --scorer hf \
    --model-dir "$(resolve_model)" --threads "$(awk -F, '{print NF}' <<<"$cores")" \
    --port "$VLLM_PORT" --served-model-name "$SERVED_NAME" --max-model-len "$MAX_MODEL_LEN"
  BACKEND_PID=$BG_PID
  BACKEND_MODEL="$SERVED_NAME"
  wait_http "http://127.0.0.1:$VLLM_PORT/health" "${VLLM_START_TIMEOUT:-1800}" "$BACKEND_PID" CPU-backend
  log "CPU backend healthy"
}

start_backend() {  # start_backend <run_dir>
  if [ "$BACKEND" = cpu ]; then start_cpu "$1"; else start_vllm "$1"; fi
}

start_gateway() {  # start_gateway <run_dir> <backend_url> <backend_model> <tokenizer_dir>
  log "decision gateway on :$PORT -> $2 ($3)"
  CUDA_VISIBLE_DEVICES="" start_bg "$1/gateway.log" "$PY" -m src.jev.server \
    --backend "$2" --backend-model "$3" --tokenizer "$4" --model-name "$MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" --temperature "$TEMPERATURE" --port "$PORT" \
    --workers "${GATEWAY_WORKERS:-1}"
  GATEWAY_PID=$BG_PID
  wait_http "http://127.0.0.1:$PORT/healthz" 300 "$GATEWAY_PID" gateway
  log "gateway healthy: POST http://127.0.0.1:$PORT/v1/systemone"
}

write_env() {  # versions and hardware next to the results
  "$PY" - "$1" "$(resolve_model)" <<PY
import json, subprocess, sys, vllm, torch, transformers
run, model_dir = sys.argv[1:]
from src.jev.decision import PROMPT_VERSION
if "$BACKEND" == "vllm":
    hw = {"gpus": subprocess.run(["nvidia-smi", "-i", "$GPUS", "--query-gpu=index,name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines(),
          "tp": $TP, "dp": $DP, "ep": $EP, "gpu_memory_utilization": $GPU_MEM_UTIL,
          "nvcc_for_jit": "$VENV/cuda13.0-jit (CUDA 13.0.88)"}
else:
    cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    hw = {"cpu": cpu, "cpu_node": $CPU_NODE, "cpu_cores": open(run + "/cpu_cores.txt").read().strip(),
          "dtype": "bfloat16", "engine": "transformers " + transformers.__version__}
json.dump({"backend": "$BACKEND", **hw, "vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
           "model": "$MODEL", "model_dir": model_dir, "lora": "${LORA:-}" or None,
           "prompt_version": PROMPT_VERSION, "temperature": float("$TEMPERATURE"),
           "max_model_len": $MAX_MODEL_LEN, "jevbench_rev": "$JEVBENCH_REV", "vllm_extra_args": "${VLLM_EXTRA_ARGS:-}",
           "vllm_env": {k: v for k, v in __import__("os").environ.items() if k.startswith("VLLM_")}},
          open(run + "/env.json", "w"), indent=2)
PY
}

bench() {  # bench <run_dir>: serial, one request at a time -- the benchmark's own protocol
  local run rc=0
  run="$(cd "$1" && pwd)"
  log "warm-up: $WARMUP untimed decisions"
  "$PY" scripts/jev_loadtest.py --endpoint "http://127.0.0.1:$PORT" --tasks "$(tasks_spec)" \
    --jevbench-dir "$JEVBENCH_DIR" --model "$MODEL_NAME" --warmup "$WARMUP" --warmup-only
  mark "$run" warmup_done
  log "JevBench: 231 public decisions, serial, typesafe adapter -> $run"
  PYTHONPATH="$PWD/$JEVBENCH_DIR" "$PY" -m jevbench.cli run \
    --tasks "$(tasks_spec)" --adapter typesafe --endpoint "http://127.0.0.1:$PORT" --key-env '' \
    --model "$MODEL_NAME" --run-label "$MODEL_NAME" \
    --cost-basis local_existing_gpu_compute_unpriced --reserve-usd 0 --cap-usd 0 \
    --results "$run/results.jsonl" --raw-dir "$run/raw" --ledger "$run/ledger.jsonl" \
    --manifest "$run/manifest.json" 2>&1 | tee "$run/bench.log" || rc=$?
  [ "$rc" = 0 ] || log "JevBench run ended early (exit $rc); the report covers what was attempted"
}

load() {  # load <run_dir>; errors at some level are reported, not fatal to the flow
  local rc=0
  log "load test: closed-loop concurrency $CONCURRENCY"
  "$PY" scripts/jev_loadtest.py --endpoint "http://127.0.0.1:$PORT" --tasks "$(tasks_spec)" \
    --jevbench-dir "$JEVBENCH_DIR" --model "$MODEL_NAME" --concurrency "$CONCURRENCY" \
    --vllm-metrics "http://127.0.0.1:$VLLM_PORT/metrics" ${LOAD_REQUESTS:+--requests "$LOAD_REQUESTS"} \
    $([ "$BACKEND" = vllm ] && echo --gpus "$NGPU" ${GPU_HOUR_USD:+--gpu-hour-usd "$GPU_HOUR_USD"}) \
    --out "$1/load.json" 2>&1 | tee "$1/load.log" || rc=$?
  [ "$rc" = 0 ] || log "load test finished with failed requests (exit $rc); see $1/load.json"
}

report() {
  "$PY" scripts/jev_report.py --run-dir "$1" --tasks "$(tasks_spec)" --jevbench-dir "$JEVBENCH_DIR" \
    --endpoint-kind "$ENDPOINT_KIND" ${COMPARE:+--compare "$COMPARE"}
}

# Wall-clock phases of a flow: `mark <run> <phase>` appends "<phase> <epoch>" and
# `timing_summary <run>` turns the marks into timing.json (seconds per phase).
mark() { echo "$2 $(date +%s.%N)" >> "$1/timing.tsv"; }

timing_summary() {
  "$PY" - "$1" <<'PY'
import json, sys
run = sys.argv[1]
marks = [(name, float(t)) for name, t in (line.split() for line in open(run + "/timing.tsv"))]
phases = {f"{a}_to_{b}_s": round(tb - ta, 2) for (a, ta), (b, tb) in zip(marks, marks[1:])}
out = {"phases_s": phases, "total_s": round(marks[-1][1] - marks[0][1], 2), "marks": dict(marks)}
json.dump(out, open(run + "/timing.json", "w"), indent=2)
print("[timing] " + ", ".join(f"{k} {v:.0f}s" for k, v in phases.items()) + f" | total {out['total_s']:.0f}s")
PY
}

orderflip() {  # orderflip <run_dir>: options reversed on every choice item, against the live server
  "$PY" scripts/jev_orderflip.py --endpoint "http://127.0.0.1:$PORT" --run-dir "$1" \
    --jevbench-dir "$JEVBENCH_DIR" --model "$MODEL_NAME" 2>&1 | tee "$1/orderflip.log"
}

new_run_dir() {
  local d="${RUN_DIR:-logs/jev/${1}_$(date +%Y%m%d-%H%M%S)}"
  mkdir -p "$d"
  echo "$d"
}

# Everything below runs from inside main(), and main is only called on the last
# line: bash parses a whole function before executing it, so editing this file
# while a long flow is running cannot corrupt the running instance.
main() {
  case "${1:-check}" in
    setup) setup ;;

    check)
      need_venv
      log "self-check (CPU)"
      CUDA_VISIBLE_DEVICES="" "$PY" scripts/jev_selfcheck.py --tokenizer "$(resolve_model)" \
        --jevbench-dir "$JEVBENCH_DIR" --max-model-len "$MAX_MODEL_LEN"
      # The real gateway, harness, load test and report against a CPU stand-in for
      # vLLM that enforces the /v1/completions contract the gateway relies on.
      run="$(new_run_dir check)"
      VLLM_PORT="$(free_port)"; PORT="$(free_port)"
      log "mock vLLM (hash scorer) on :$VLLM_PORT"
      start_bg "$run/vllm.log" "$PY" scripts/jev_cpu_backend.py --scorer hash --port "$VLLM_PORT" \
        --served-model-name "$SERVED_NAME" --max-model-len "$MAX_MODEL_LEN"
      wait_http "http://127.0.0.1:$VLLM_PORT/health" 60 "$BG_PID" mock-vLLM
      start_gateway "$run" "http://127.0.0.1:$VLLM_PORT" "$SERVED_NAME" "$(resolve_model)"
      WARMUP=16 bench "$run"
      CONCURRENCY="1,4,16" LOAD_REQUESTS="" load "$run"
      report "$run" > /dev/null
      "$PY" - "$run/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
c, levels = r["coverage"], r["load"]["levels"]
ok = (c["attempted"] == c["ok"] == c["valid"] == c["tasks"] == 231
      and all(not s["errors"] for s in levels) and r["head_to_head"])
print(f"  [{'PASS' if ok else 'FAIL'}] {c['ok']}/{c['tasks']} decisions ok, {c['valid']} valid; "
      f"load levels {[s['concurrency'] for s in levels]} without errors; "
      f"report compares against {list(r['head_to_head'])}")
sys.exit(0 if ok else 4)
PY
      log "CHECK PASSED (mock backend: accuracies in $run are noise by construction)"
      ;;

    gpu-check) gpu_check ;;

    train)
      need_venv
      "$PY" train_jev.py "${@:2}"
      ;;

    check-train)
      need_venv
      CUDA_VISIBLE_DEVICES="" "$PY" scripts/jev_train_smoketest.py "${@:2}"
      ;;

    serve)
      need_venv
      [ "$BACKEND" = cpu ] || gpu_check
      run="$(new_run_dir "serve_$BACKEND")"
      start_backend "$run"
      start_gateway "$run" "http://127.0.0.1:$VLLM_PORT" "$BACKEND_MODEL" "$(resolve_model)"
      write_env "$run"
      log "serving; logs in $run. Ctrl-C stops both servers."
      wait "$BACKEND_PID"
      ;;

    bench)
      need_venv
      run="$(new_run_dir bench)"
      bench "$run"
      report "$run"
      ;;

    load)
      need_venv
      run="${2:-$(new_run_dir load)}"
      load "$run"
      if [ -f "$run/results.jsonl" ]; then
        report "$run" > /dev/null
        log "report updated: $run/report.md"
      fi
      ;;

    flow)
      need_venv
      [ "$BACKEND" = cpu ] || gpu_check
      run="$(new_run_dir "flow_${BACKEND}_${MODEL_NAME}")"
      log "flow -> $run"
      mark "$run" start
      start_backend "$run"
      mark "$run" backend_healthy            # model load (+ compile / graph capture on vLLM)
      start_gateway "$run" "http://127.0.0.1:$VLLM_PORT" "$BACKEND_MODEL" "$(resolve_model)"
      mark "$run" gateway_healthy
      write_env "$run"
      bench "$run"                           # marks warmup_done inside
      mark "$run" bench_done
      log "order-flip: choice options reversed"
      orderflip "$run"
      mark "$run" orderflip_done
      load "$run"
      mark "$run" load_done
      report "$run"
      mark "$run" report_done
      timing_summary "$run"
      date '+%F %T' > "$run/FLOW_DONE"
      log "flow complete: $run/report.md"
      ;;

    *) sed -n '2,23p' "$0"; exit 1 ;;
  esac
}

main "$@"; exit $?
