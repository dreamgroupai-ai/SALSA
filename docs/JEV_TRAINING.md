# Training a SALSA decision model on your own data

This guide trains a LoRA adapter so that an open LLM answers *your* typed decisions:
routing, policy checks, triage, extraction, rating. It is served as a Jev-style API
afterwards. The method is SALSA's single-token objective applied to typed questions,
and training uses the **exact prompt and readout the server uses**, so nothing changes
between training and deployment.

```bash
./run_jev.sh setup                                   # once: environment (vLLM, transformers, peft)
./run_jev.sh train --model Qwen/Qwen3.8-27B \
    --train data/decisions.jsonl --out outputs/my-decider \
    --exclude-overlap datasets/jevbench/datasets/public/easy.jsonl,datasets/jevbench/datasets/public/original.jsonl,datasets/jevbench/datasets/public/hard.jsonl
LORA=outputs/my-decider/best ./run_jev.sh serve      # serve it (vLLM)
```

`./run_jev.sh train …` is `python train_jev.py …` inside the project environment. For
several GPUs, use `accelerate launch train_jev.py …` (DDP).

---

## 1. Data format

JSONL, one decision per line. This is JevBench's record format, so JevBench-style files
load unchanged. [`examples/jev_decisions_example.jsonl`](../examples/jev_decisions_example.jsonl)
has 30 complete examples.

```json
{"id": "t-0007",
 "state": "Policy: refunds need a receipt and a purchase within 30 days. Bought 12 days ago, no receipt. Issue a refund.",
 "question": {"type": "noul",
              "instructions": "Under the stated policy, is the refund permitted? Treat unproved conditions as not satisfied.",
              "criteria": {"true": "Every required condition is established", "false": "A condition is missing"}},
 "expected": "no",
 "family": "policy"}
```

| field | required | meaning |
|---|---|---|
| `state` | yes | text, or any JSON object (shown to the model as indented JSON) |
| `question.type` | yes | `noul` (yes/no), `choice` (one of named options), `score` (ordered level) |
| `question.instructions` | yes | the decision rule, edge cases included |
| `question.criteria` | choice, score; optional for noul | `{key: description}` / `[level 0, level 1, …]` / `{"true": …, "false": …}` |
| `expected` | yes, unless `probs` is given | the gold option key, `"yes"`/`"no"`, or the score level (an integer) |
| `probs` | no | a soft target over the answer keys, e.g. annotator votes `{"refund": 0.7, "other": 0.3}`; overrides `expected` |
| `weight` | no | per-item loss weight (default 1) |
| `group` | no | items sharing a group (e.g. paraphrases) stay on one side of the train/validation split |
| `family` | no | reported separately in the evaluation |
| `id` | no | used in reports; defaults to `file:line` |

Limits are the server's own: 2–26 options per choice question and 2–10 score levels.
Prompts longer than `--max-length` (default 4,096 tokens) are **skipped, never
truncated**, because a cut-off state is a different decision. Invalid lines are
reported and skipped.

**Writing good training items** follows the same rules as writing good questions for the
API ([DECISION_API.md §4](DECISION_API.md)):

- Put the governing rule in the state or the instructions.
- Use meaningful option keys.
- Include an explicit `other` / `none` option where nothing may apply.
- Precompute dates and arithmetic.

The model learns to apply the rule *it is shown*. It does not memorise a rule that only
exists in your head.

## 2. What the training does

1. **Prompt.** Each item is rendered by `src/jev/decision.PromptBuilder`, the code the
   server runs. It uses the model's chat template with thinking off, the state, the
   question, the options as `A.`, `B.`, … (or `yes`/`no`, or level digits), and ends at
   the answer anchor `<ANSWER>\n`.
2. **Loss.** This is SALSA's single-token objective. One forward pass takes the logits at
   the answer position, restricted to *this item's* class tokens (2–26 of them), and
   computes cross-entropy against the gold option, or against the soft target if `probs`
   is given. Only the answer positions pass through the output head, so memory does not
   grow with vocabulary × sequence length.
3. **Option shuffling** (on by default; `--no-shuffle-options` turns it off). Every time a
   choice item is drawn, its options are re-ordered, so the right answer's letter moves and
   the target follows it. Off-the-shelf readers change 7–31% of their choice answers when
   the options are reversed ([JEVBENCH_RESULTS.md](JEVBENCH_RESULTS.md)). Shuffling teaches
   the option, not the position. Score levels keep their order, since the order is their
   meaning.
4. **LoRA on all linear layers** (`--lora-targets all-linear`, r=16, α=32). On hybrid
   models such as Qwen3.5/3.8 this includes the linear-attention projections.
5. **Selection.** Validation runs after every epoch (accuracy, NLL, Brier, ECE), and the
   adapter with the lowest validation NLL is kept as `best/`.

## 3. Avoiding benchmark leakage

If you will ever report a JevBench (or other benchmark) number for the trained model,
pass that benchmark's files to `--exclude-overlap`. Items whose normalised state +
instructions match a benchmark item are dropped and counted, the same exact-match check
JevBench applies to submitted training data. Training on benchmark items makes the
benchmark score meaningless. It also shows up as a public-to-sealed accuracy gap, which
JevBench penalises.

A large public candidate source is **BEV Decision Mix** (`avbiswas/bev-decision` on
Hugging Face):

- ~104k training decisions in its default configuration, plus `hard_50k` and a
  `numeric_temporal` set;
- the same three question types;
- its licence is listed as *unknown* — check it before use;
- its overlap with JevBench has not been measured — use `--exclude-overlap`.

## 4. Options

| option | default | |
|---|---|---|
| `--model` | — | base model (Hugging Face id or directory); the server must use the same base |
| `--train`, `--val` | — / 10% of train | JSONL; `--train a.jsonl,b.jsonl` concatenates |
| `--val-fraction` | 0.1 | split by `group` (or `id`), deterministic |
| `--epochs`, `--max-steps` | 2, — | |
| `--lr` | 1e-4 | |
| `--batch-size`, `--grad-accum` | 4, 4 | per device; effective batch = batch × accum × GPUs |
| `--lora-r`, `--lora-alpha`, `--lora-dropout` | 16, 32, 0.05 | |
| `--gradient-checkpointing` | off | for long prompts or large models |
| `--dtype` | auto | bf16 on GPU, fp32 on CPU |
| `--merge-to DIR` | — | also write base + adapter merged into one model directory |
| `--report-to` | none | `wandb`, `tensorboard`, … |

## 5. Outputs and serving

```
outputs/my-decider/
  best/                 the adapter with the lowest validation NLL (PEFT format) + tokenizer + salsa_jev.json
  final/                the adapter after the last step
  checkpoints/          Trainer checkpoints (resume with the HF Trainer as usual)
  train_report.json     validation metrics before and after training, per type and per family
```

Serve the adapter with vLLM:

```bash
MODEL=Qwen/Qwen3.8-27B LORA=outputs/my-decider/best ./run_jev.sh serve
```

or serve the merged model (`--merge-to outputs/my-decider-merged`) as a plain model, on
vLLM or on CPU:

```bash
MODEL=outputs/my-decider-merged ./run_jev.sh serve
BACKEND=cpu MODEL=outputs/my-decider-merged ./run_jev.sh serve
```

Then evaluate exactly as for a base model. `./run_jev.sh bench` runs the JevBench
public set, and `scripts/jev_compare.py` puts the result beside other systems.

## 6. Verified behaviour

| check | how | result |
|---|---|---|
| train → merge → serve, CPU | `./run_jev.sh check-train` (tiny random copy of Qwen3; ~2 min) | served probabilities equal the trainer's readout to 1.5e-7 on all 30 example decisions; shuffling moves the gold letter and the target follows |

This is a plumbing and consistency check on a toy set, not a quality claim; the
example set is far too small to generalise from.

`check-train` serves the merged model; serving the LoRA adapter itself through vLLM
(`LORA=…`) is not part of it. For hybrid checkpoints (Qwen3.5/3.8), `--merge-to` gives a
plain model directory that serves without any adapter-name mapping.
