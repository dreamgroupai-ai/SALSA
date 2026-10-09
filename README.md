# SALSA: Single-pass Autoregressive LLM Structured Classification
> This repository accompanies the paper: [SALSA: Single-pass Autoregressive LLM Structured Classification](https://arxiv.org/abs/2510.22691)

SALSA is a lightweight fine-tuning method that turns Large Language Models into efficient text classifiers. Instead of generating full text responses, SALSA trains the model to predict a **single classification token** per input, computing loss only on that one token position. This makes training fast and memory-efficient while achieving strong results across NLP benchmarks and competition tasks.

## SALSA as a Typed Decision Engine

[SALSA: Single-pass Autoregressive LLM Structured Classification](https://arxiv.org/abs/2510.22691) introduced a framework for efficient structured classification with generative LLMs. Rather than generating and parsing free-form answers, SALSA maps each admissible class to a **single token** and reads the probabilities of these tokens at **one answer position, in one forward pass**. The same class-token formulation supports task-specific fine-tuning with a classification loss over the allowed outcomes.

**Typed decision-making is a subclass of classification.** A typed question restricts its answer to a finite set of valid outcomes, whether a binary yes/no decision, a choice among named options, or an ordinal rating. Each is a classification task with an explicitly defined label space. The resulting distribution over valid answers is precisely the class distribution computed by SALSA.

**Typed decisions are therefore already covered by SALSA's original methodology.** No new modeling paradigm, training objective, or inference algorithm is necessary. What this repository adds is the **serving envelope**: request parsing, per-request label-to-token mapping, probability extraction, API compatibility, and deployment tools that make the existing SALSA approach usable as a decision service.

### Serving SALSA as a typed decision API

The serving layer provides:

- **Request-defined label spaces.** An API request specifies the options for a binary decision, categorical choice (2–26 options), or ordinal rating (2–10 levels). The serving layer maps these to SALSA's single-token class representation; the task remains classification.
- **Single-pass probability readout.** SALSA extracts the logits of the allowed class tokens at one answer position and normalizes them into a distribution. No free-text answer generation or separate forward pass for each option is required.
- **Jev-compatible interface.** The `POST /v1/systemone` endpoint supports existing Jev clients and the [JevBench](https://github.com/fstandhartinger/jevbench) harness. Jev compatibility is an integration feature, not a change to SALSA's classification method.
- **Efficient inference backends.** vLLM 0.31 serves class-token probabilities on GPU, while a Transformers backend supports CPU inference. The implementation covers dense models and hybrid linear-attention models (Qwen3.5/3.8).
- **Optional task-specific training.** Applications can fine-tune with SALSA's existing single-token classification loss over each example's options. The pipeline adds option-order shuffling and alignment between training and serving prompts.

The underlying algorithm remains **SALSA**. The API makes it available for routing, moderation, policy checks, triage, adequacy assessment, and ordinal scoring, all of which are constrained classification tasks.

### Zero-shot results on JevBench (public items)

To test SALSA's existing classification method in a typed-decision setting, we evaluated **Qwen3.8-27B without any fine-tuning**, served through vLLM and scored by JevBench's own harness on **231 public items**.

| System | Accuracy | Hard (111) | Latency P50 / P95 |
|---|---:|---:|---|
| **SALSA, Qwen3.8-27B, zero-shot** | **87.4%** | **74.8%** | **34 / 155 ms** (1× B200, localhost) |
| Jev 1.13.0 (TypeSafe AI) | 86.6% | 73.0% | 652 / 722 ms (production API, over the internet) |
| [LLM-as-Jev](https://arxiv.org/abs/2610.02076), Qwen3.5-4B, training-free / LoRA | 81.4% / 84.0% | 64.9% / 68.5% | 194 ms median (fp32, RTX A5500) |
| [tic-top/llm2jev](https://github.com/tic-top/llm2jev), Qwen3.8-27B | 84.0% | 70.3% | — |
| [LLM2Jev](https://github.com/Yinsongxu/LLM2Jev), Qwen3.5-4B | 76.2% | — | 48 / 346 ms (hardware not stated) |
| [AnyJev](https://arxiv.org/abs/2610.00831), best raw (Qwen3-32B) | 79.8% (213 items) | — | — |

Other systems' numbers are as reported by their authors. Jev's accuracy comes from JevBench's published per-item outcomes. The latency measurements use different hardware and deployment environments and are **not controlled head-to-head latency comparisons**.

- **Against Jev 1.13.0:** SALSA's accuracy difference is +0.9 percentage points (95% CI −3.0 to +4.3; McNemar p = 0.82), a statistical tie.
- **Calibration and robustness:** ECE is 0.052 with no calibration fit (15 equal-mass bins, AnyJev's protocol); 7.2% of choice answers change when the options are reversed.
- **Throughput:** 22 decisions/s per B200 serially, and 273/s at concurrency 64 across four B200 replicas.

These results show that **SALSA's original single-pass classification approach can provide competitive typed decision performance without a new decision-model architecture or task-specific fine-tuning**. They cover public items only, not an official JevBench Score. Full tables, confidence intervals, calibration, order-flip evaluations, run times, and repeatability are available in **[docs/JEVBENCH_RESULTS.md](docs/JEVBENCH_RESULTS.md)**.

### Quick start: decision API

```bash
./run_jev.sh setup     # isolated vLLM 0.31 environment + pinned JevBench checkout
./run_jev.sh check     # CPU only: full serve -> benchmark -> report path against a mock backend
./run_jev.sh serve     # Qwen3.8-27B as a /v1/systemone decision API (GPU 0)

curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Do not cancel my membership. Please return the duplicate charge.",
  "questions": {"intent": {"type": "choice",
    "instructions": "Select the primary requested action.",
    "criteria": {"cancel": "End a subscription", "refund": "Return money already charged", "other": "None of these"}}}}'

./run_jev.sh flow      # JevBench public run + latency, throughput, order-flip, run time, report

# train on your own decisions, then serve the adapter
./run_jev.sh train --model Qwen/Qwen3.8-27B --train my_decisions.jsonl --out outputs/my-decider
LORA=outputs/my-decider/best ./run_jev.sh serve
```

| Guide | |
|---|---|
| [docs/DECISION_API.md](docs/DECISION_API.md) | Starting the service, request/response format, writing good questions, reading the probabilities |
| [docs/JEV_TRAINING.md](docs/JEV_TRAINING.md) | Training a decision adapter on your own dataset: format, method, leakage guard, serving |
| [docs/JEVBENCH.md](docs/JEVBENCH.md) | Design decisions, measurement protocol, CPU backend |
| [docs/JEVBENCH_RESULTS.md](docs/JEVBENCH_RESULTS.md) | GPU results in the LLM2Jev and AnyJev layouts |

*Jev is TypeSafe AI's model, and JevBench is Benchmark Heaven's benchmark. This project is not affiliated with either. It implements Jev's public API shape and runs JevBench's public items with JevBench's own harness.*

## Benchmark Results
SALSA ranks **#7 overall** on the [GLUE leaderboard](https://gluebenchmark.com/leaderboard/) and holds **#1 on 4 individual tasks**:

| Task | Score | Rank |
|------|-------|------|
| RTE | 94.8 | **#1** |
| MNLI-m | 92.7 | **#1** |
| STS-B | 93.8 / 93.6 | **#1** |
| SST-2 | 97.9 | **#1** (tied) |

SALSA also achieves state-of-the-art results on additional benchmarks:

| Dataset | Score | Previous SOTA |
|---------|-------|---------------|
| [AG News](https://huggingface.co/datasets/SetFit/ag_news) | 95.9 | 95.5 (XLNet) |
| [IMDb](https://huggingface.co/datasets/SetFit/imdb) | 97.6 | 96.8 (XLNet) |
| [Yelp-5](https://huggingface.co/datasets/SetFit/yelp_review_full) | 74.2 | 72.9 (XLNet) |
| [MedNLI](https://huggingface.co/datasets/presencesw/mednli) | 91.3 | 90.2 (GatorTron) |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 74.1 | 73.6 (GPT-4) |
| [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) | 71.8 | 70.4 (BERT-MRP) |

## How SALSA Works
1. **Prompt formatting** -- Each sample is converted into a chat conversation: the user message contains the text to classify, and the assistant response contains a single label token (e.g., `"0"` or `"1"`).

2. **Single-token loss** -- During training, loss is computed only on the label token position, not the entire sequence. The logits are sliced down to K target tokens (where K = number of classes) and cross-entropy is computed over that K-dimensional softmax.

3. **LoRA fine-tuning** -- Only a small set of adapter weights are trained (via [LoRA](https://arxiv.org/abs/2106.09685)), keeping the base model frozen. This typically trains <0.1% of total parameters.

4. **Single-token inference** -- At inference time, the model generates one token. The predicted class is whichever target token has the highest logit.

## Table of Contents
- [SALSA as a Typed Decision Engine](#salsa-as-a-typed-decision-engine)

- [Benchmark Results](#benchmark-results)

- [Requirements](#requirements)

- [Installation](#installation)

- [Quick Start](#quick-start)

- [Adding Your Own Dataset](#adding-your-own-dataset)

- [Configuration Reference](#configuration-reference)

- [Training](#training)

- [Inference](#inference)

- [Evaluation](#evaluation)

- [Supported Datasets](#supported-datasets)

- [Advanced Features](#advanced-features)

## Requirements
- Python 3.12

- NVIDIA GPU with CUDA support

- \~8 GB VRAM minimum for the default model (Qwen3-0.6B with LoRA)

- Multi-GPU recommended for larger models (8B+)

- Decision API (`run_jev.sh`): an NVIDIA driver with CUDA 13.0 support (R580+) for vLLM 0.31.

  `./run_jev.sh setup` creates its own environment (`.venv-vllm`), separate from the Poetry one.

  The CPU backend needs no GPU.

## Installation
```bash

# Install dependencies via Poetry

poetry install

# Set environment variables

export HuggingfaceToken="hf_your_token_here"   # optional if models are cached locally

export WANDB_API_KEY="your_key_here"            # or use WANDB_MODE=offline

```

## Quick Start
Train a sentiment classifier on SST-2 in one command:

```bash

python run.py --config-name sst2

```

This downloads the Qwen3-0.6B model, loads the GLUE/SST-2 dataset from HuggingFace, applies LoRA, and trains for 6 epochs. Checkpoints are saved to `outputs/sst2_checkpoint/`.

Evaluate the trained model:

```bash

python run_eval.py --config-name sst2 \\

    model.load_checkpoint=outputs/sst2_checkpoint/checkpoint-5000

```

## Adding Your Own Dataset
### Step 1: Prepare your data
Place your data as parquet files in a directory:

```

datasets/my_task/

├── train.parquet        # must have a 'label' column

├── validation.parquet   # must have a 'label' column

└── test.parquet         # optional; can omit 'label' for inference-only

```

### Step 2: Create a config file
Create `configs/my_task.yaml`. Start from an existing config that matches your task type:

**Binary classification:**

```yaml

defaults:

  - base_config.yaml

  - _self_

dataset:

  name: "unused"                       # ignored when local_path is set

  local_path: "datasets/my_task"

  val_name: "validation"

  test_name: "test"

  labels: [0, 1]                       # values in your 'label' column

  fields:                              # map column names -> prompt placeholders

    text: "review_text"                # renames 'review_text' column to 'text'

logging:

  save_steps:

    eval_step: 500

output:

  checkpoint_path: "outputs/my_task"

classifier:

  main_prompt: |

    Classify the sentiment of the following review:

    <TEXT>

    {text}

    </TEXT>

    Respond with 0 for negative or 1 for positive.

    <ANSWER>

  answer_prompt: |

    <ANSWER>{label}</ANSWER>

  options:

    label: ["0", "1"]                  # tokens the model predicts (must be single tokens)

  prefix: <ANSWER>

```

**Multi-class with dual input:**

```yaml

defaults:

  - base_config.yaml

  - _self_

dataset:

  name: "unused"

  local_path: "datasets/my_nli_task"

  val_name: "validation"

  test_name: ""

  labels: [0, 1, 2]

  fields:

    text1: "premise"                   # renames 'premise' -> 'text1'

    text2: "hypothesis"                # renames 'hypothesis' -> 'text2'

logging:

  save_steps:

    eval_step: 1000

output:

  checkpoint_path: "outputs/my_nli"

classifier:

  main_prompt: |

    Given the premise: {text1}

    And the hypothesis: {text2}

    Is the hypothesis entailed, neutral, or contradicted?

    <ANSWER>

  answer_prompt: |

    <ANSWER>{label}</ANSWER>

  options:

    label: ["0", "1", "2"]

  prefix: <ANSWER>

```

### Step 3: Train
```bash

python run.py --config-name my_task

```

Override any config value from the command line:

```bash

python run.py --config-name my_task \\

    training.epochs=3 \\

    training.learning_rate=5e-5 \\

    model.id="Qwen/Qwen3-8B"

```

### Key concepts
- **`labels`** -- The values in your dataset's `label` column. These get mapped to the classifier `options.label` tokens in order. For example, `labels: [0, 1]` with `options.label: ["0", "1"]` maps dataset label `0` to token `"0"`.

- **`fields`** -- Maps your dataset's column names to the placeholders used in `main_prompt`. If your column is already named the same as the placeholder, you can omit the mapping. For example, if your dataset has a `code` column and your prompt uses `{code}`, no field mapping is needed.

- **`options.label`** -- The actual tokens the model will predict. Each must be a single token in the model's vocabulary. Common choices: `["0", "1"]`, `["A", "B", "C"]`, `["negative", "positive"]`.

- **`main_prompt`** -- The user message template. Use `{placeholder}` syntax to insert dataset fields. The placeholders must match either your column names directly or the keys in `fields`.

- **`answer_prompt`** -- The assistant response template. Must contain `{label}` which gets replaced with the mapped label token.

## Configuration Reference
SALSA uses [Hydra](https://hydra.cc/) for configuration. Each dataset has its own YAML config that inherits from `configs/base_config.yaml`, which pulls in defaults from subdirectories.

### Config structure
```

configs/

├── base_config.yaml          # wires up all default sub-configs

├── model/default.yaml        # model ID, cache dir

├── training/default.yaml     # epochs, learning rate, batch size

├── lora/default.yaml         # LoRA rank, alpha, dropout, target modules

├── dataset/default.yaml      # dataset name, splits, labels, fields

├── classifier/default.yaml   # prompt templates, label options

├── logging/default.yaml      # eval frequency, W&B project

├── output/default.yaml       # checkpoint path

├── inference/default.yaml    # vLLM tensor parallelism, batch size

└── <dataset>.yaml            # per-dataset overrides (e.g., rte.yaml, task13a8B.yaml)

```

### Training defaults
| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 6 | Number of training epochs |
| `training.learning_rate` | 1e-4 | Learning rate |
| `training.gradient_accumulation_steps` | 25 | Steps before weight update |
| `training.max_seq_length` | 1024 | Max tokens per sequence (left-truncated) |
| `training.weight_decay` | 1e-4 | AdamW weight decay |
| `training.seed` | 42 | Random seed |
| `training.save_total_limit` | 2 | Max checkpoints to keep |
| `training.resume` | true | Auto-resume from last checkpoint |
| `training.deepspeed_config` | null | DeepSpeed config file (null = disabled) |

### LoRA defaults
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | 8 | LoRA rank |
| `lora.alpha` | 16 | LoRA scaling factor |
| `lora.dropout` | 0.1 | LoRA dropout |
| `lora.target_modules` | q/k/v/o/gate/up/down_proj | Modules to apply LoRA to |

### Model defaults
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.id` | Qwen/Qwen3-0.6B | HuggingFace model ID |
| `model.cache_dir` | cache_model | Local model cache directory |
| `model.load_checkpoint` | null | PEFT checkpoint path for warm-start or eval |

For large models, override these in your task config (see the `task13a8B.yaml` configs for examples with DeepSpeed ZeRO-3, larger LoRA rank, and longer sequences).

## Training
### Single GPU
```bash

python run.py --config-name rte

```

### Multi-GPU (DDP via Accelerate)
```bash

accelerate launch --num_processes=4 run.py --config-name rte

```

### Multi-GPU with DeepSpeed ZeRO-3
For large models (8B+), enable DeepSpeed in your config:

```yaml

training:

  deepspeed_config: "ds_config_zero3.json"

  max_seq_length: 8192

  gradient_accumulation_steps: 50

lora:

  r: 16

  alpha: 32

  dropout: 0.05

```

Then launch with Accelerate:

```bash

accelerate launch --num_processes=8 run.py --config-name task13a8B

```

### Resume training
Training auto-resumes from the last checkpoint by default:

```bash

# Resumes automatically if checkpoints exist

python run.py --config-name rte

# Force fresh start

python run.py --config-name rte training.resume=false

```

### Warm-start from a checkpoint
Train with different hyperparameters starting from existing LoRA weights:

```bash

python run.py --config-name rte \\

    model.load_checkpoint=outputs/previous_run/checkpoint-500 \\

    training.learning_rate=5e-5

```

## Inference
### Accelerate-based inference
```bash

# Single GPU

python run_inference.py --config-name rte \\

    model.load_checkpoint=/path/to/checkpoint

# Multi-GPU

accelerate launch run_inference.py --config-name rte \\

    model.load_checkpoint=/path/to/checkpoint

```

Output is a CSV with `ID,label` columns.

### vLLM inference (high throughput)
For large test sets, vLLM provides significantly higher throughput using single-token constrained generation:

```bash

python run_inference_vllm.py --config-name task13a8B \\

    model.load_checkpoint=/path/to/checkpoint \\

    inference.tensor_parallel_size=4

```

| vLLM option | Default | Description |
|-------------|---------|-------------|
| `inference.tensor_parallel_size` | Auto | GPUs for tensor parallelism |
| `inference.batch_size` | 32 | Batch size |
| `inference.gpu_memory_utilization` | 0.9 | GPU memory fraction |

### Zero-shot inference (no adapter)
Establish a baseline without any fine-tuning:

```bash

python run_inference_vllm.py --config-name task13a32B \\

    +output.submission_path=submission_zeroshot.csv

```

### When to use which
| Scenario | Script |
|----------|--------|
| Large test sets (>10K samples) | `run_inference_vllm.py` |
| Limited GPU memory | `run_inference.py` |
| Maximum throughput | `run_inference_vllm.py` |
| Debugging / small batches | `run_inference.py` |

## Evaluation
Evaluate a model on the validation or test set with computed metrics (accuracy, ROC-AUC, F1):

```bash

# Evaluate base model (no fine-tuning)

python run_eval.py --config-name rte

# Evaluate fine-tuned checkpoint

python run_eval.py --config-name rte \\

    model.load_checkpoint=outputs/rte_checkpoint/checkpoint-500

```

Results are printed to console and saved as JSON.

## Supported Datasets
### GLUE benchmarks
| Config | Dataset | Task | Classes |
|--------|---------|------|---------|
| `sst2` | GLUE/SST-2 | Sentiment | 2 |
| `cola` | GLUE/CoLA | Linguistic acceptability | 2 |
| `qnli` | GLUE/QNLI | Question NLI | 2 |
| `mnli` | GLUE/MNLI | Multi-genre NLI | 3 |
| `qqp` | GLUE/QQP | Paraphrase detection | 2 |
| `rte` | GLUE/RTE (via SetFit) | Textual entailment | 2 |
| `mrpc` | GLUE/MRPC (via SetFit) | Paraphrase detection | 2 |
| `stsb` | GLUE/STS-B | Semantic similarity | 6 |
| `wnli` | GLUE/WNLI (via Gretel) | Winograd NLI | 2 |

### Other NLP tasks
| Config | Dataset | Task | Classes |
|--------|---------|------|---------|
| `agnews` | AG News (via SetFit) | Topic classification | 4 |
| `imdb` | IMDB (via SetFit) | Sentiment | 2 |
| `yelp5` | Yelp Reviews (via SetFit) | Sentiment | 5 |
| `hatexplain` | HateXplain | Hate speech detection | 3 |
| `mednli` | MedNLI | Medical NLI | 3 |
| `medmcqa` | MedMCQA | Medical QA (4-choice) | 4 |

### SemEval-2026 Task 13A (machine-generated code detection)
| Config | Model | Notes |
|--------|-------|-------|
| `task13a8B` | Qwen3-8B | DeepSpeed ZeRO-3, balanced sampling |
| `task13a32B` | Qwen3-32B | DeepSpeed ZeRO-3, balanced sampling |
| `task13a25_qwen_72` | Qwen2.5-72B-Instruct | Zero-shot validation |

These configs use larger LoRA rank (16), longer sequences (8192), and DeepSpeed ZeRO-3 for multi-GPU training.

## Advanced Features
### Sequence length and truncation
Sequences longer than `max_seq_length` are truncated from the **left** (beginning), preserving the label token at the end. The default is 1024 tokens, sufficient for most NLP tasks. For longer inputs (e.g., code files), increase it in your config:

```yaml

training:

  max_seq_length: 8192

```

Higher values require more GPU memory.

### Balanced sampling
For datasets with class imbalance, enable stratified sampling:

```yaml

dataset:

  balanced_sampling:

    enabled: true

    max_repetitions: 3

    balance_columns: ["original_label", "language"]

```

This ensures equal representation across all combinations of the specified columns. Each sample appears at most `max_repetitions` times per epoch.

### Test set inference during training
When `dataset.test_name` points to a test split, the trainer automatically generates predictions at each checkpoint save. If the test set has labels, F1 scores are logged to W&B as `test/f1`.

Submissions are saved to:

- `<checkpoint_path>/checkpoint-<step>/submission.csv` (at each checkpoint)

- `<checkpoint_path>/final_submission.csv` (after training)

### Local dataset loading
Load datasets from local parquet files instead of HuggingFace:

```yaml

dataset:

  name: "unused"

  local_path: "datasets/my_data"

  val_name: "validation"

  test_name: "test"

```

Expected file structure:

```

datasets/my_data/

├── train.parquet

├── validation.parquet

└── test.parquet

```

The `name` field is ignored when `local_path` is set, but must still be present in the config.

## Project Structure
```

.

├── run.py                     # Training entry point (Hydra)

├── run_eval.py                # Evaluation entry point

├── run_inference.py           # Accelerate-based inference

├── run_inference_vllm.py      # vLLM-based inference (high throughput)

├── run_jev.sh                 # Decision API: setup, check, serve, bench, load, flow, train

├── train_jev.py               # Train a decision adapter on your own typed decisions

├── src/

│   ├── salsa_trainer.py       # SALSATrainer, TaskTrainer, BalancedSampler, metrics

│   ├── manage_datasets.py     # SalsaDatasetFactory (loading, formatting, tokenization)

│   ├── adapter_utils.py       # LoRA adapter key remapping for Qwen3.5

│   └── jev/                   # SALSA as a Jev-style decision engine

│       ├── decision.py        # typed question -> prompt, class tokens, readout, typed answer

│       ├── server.py          # /v1/systemone gateway over vLLM

│       └── train.py           # decision dataset, option shuffling, single-token loss, evaluation

├── scripts/                   # jev_* tools: self-check, CPU backend, load test, order-flip, reports

├── examples/                  # jev_decisions_example.jsonl: training-format example

├── docs/                      # decision API, training, JevBench design and results

├── .claude/skills/            # Claude Code skill for using the decision API

├── configs/                   # Hydra configuration files

│   ├── base_config.yaml       # Default config composition

│   ├── <dataset>.yaml         # Per-dataset configs

│   └── <group>/default.yaml   # Default values for each config group

├── ds_config_zero2.json       # DeepSpeed ZeRO-2 config

├── ds_config_zero3.json       # DeepSpeed ZeRO-3 config

├── ds_config_zero3_moe.json   # DeepSpeed ZeRO-3 config for MoE models

└── pyproject.toml             # Dependencies (Poetry)

```

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
