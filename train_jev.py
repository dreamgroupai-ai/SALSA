#!/usr/bin/env python3
"""Train a SALSA decision adapter (Jev-style typed decisions) on your own dataset.

    python train_jev.py --model Qwen/Qwen3.8-27B --train data/train.jsonl --out outputs/jev-adapter
    accelerate launch train_jev.py ...                       # multi-GPU (DDP)

Data format, method and serving: docs/JEV_TRAINING.md.  The result serves with
`LORA=outputs/jev-adapter/best ./run_jev.sh serve` (vLLM), or -- after `--merge-to` --
as a plain model directory with `MODEL=<dir>` (vLLM or the CPU backend).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("data")
    io.add_argument("--model", required=True, help="base model: Hugging Face id or local directory")
    io.add_argument("--train", required=True, help="decision JSONL (comma-separated for several files)")
    io.add_argument("--val", default=None, help="validation JSONL; default: --val-fraction of --train")
    io.add_argument("--val-fraction", type=float, default=0.1)
    io.add_argument("--exclude-overlap", default=None,
                    help="comma-separated benchmark JSONL files (e.g. JevBench public); matching items are dropped")
    io.add_argument("--out", required=True, help="output directory")
    io.add_argument("--max-length", type=int, default=4096, help="longer prompts are skipped, never truncated")
    io.add_argument("--no-shuffle-options", action="store_true", help="keep choice options in the given order")

    tr = ap.add_argument_group("training")
    tr.add_argument("--epochs", type=float, default=2.0)
    tr.add_argument("--max-steps", type=int, default=-1)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--batch-size", type=int, default=4, help="per device")
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--warmup-ratio", type=float, default=0.03)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--lora-dropout", type=float, default=0.05)
    tr.add_argument("--lora-targets", default="all-linear",
                    help="PEFT target_modules: 'all-linear' (default) or comma-separated module names")
    tr.add_argument("--gradient-checkpointing", action="store_true")
    tr.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto",
                    help="auto: bfloat16 on GPU, float32 on CPU")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--logging-steps", type=int, default=10)
    tr.add_argument("--save-steps", type=int, default=0, help="0: checkpoint every epoch")
    tr.add_argument("--eval-batch-size", type=int, default=8)
    tr.add_argument("--report-to", default="none", help="e.g. wandb, tensorboard")
    tr.add_argument("--trust-remote-code", action="store_true")

    ex = ap.add_argument_group("export")
    ex.add_argument("--merge-to", default=None,
                    help="also save base + adapter merged into one model directory (servable with MODEL=<dir>)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

    from src.jev.decision import PromptBuilder
    from src.jev.train import (DecisionCollator, DecisionDataset, build_trainer, drop_benchmark_overlap,
                               evaluate, load_records, metadata, split_records)

    set_seed(a.seed)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    say = (lambda *x: print("[train_jev]", *x, flush=True)) if rank0 else (lambda *x: None)

    # ---- data ---------------------------------------------------------------------------- #
    records, problems = [], []
    for path in a.train.split(","):
        r, p = load_records(path.strip())
        records += r
        problems += p
    val_records = None
    if a.val:
        val_records, p = load_records(a.val)
        problems += p
    if problems:
        say(f"{len(problems)} invalid lines skipped; first: {problems[0]}")
    if a.exclude_overlap:
        bench = [p.strip() for p in a.exclude_overlap.split(",") if p.strip()]
        records, dropped = drop_benchmark_overlap(records, bench)
        if val_records is not None:
            val_records, dropped_val = drop_benchmark_overlap(val_records, bench)
            dropped += dropped_val
        say(f"benchmark overlap: dropped {len(dropped)} items")
    if val_records is None:
        records, val_records = split_records(records, a.val_fraction, a.seed)
    if not records:
        sys.exit("no training decisions left")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    builder = PromptBuilder(tok)
    train_ds = DecisionDataset(records, builder, a.max_length, shuffle_options=not a.no_shuffle_options, seed=a.seed)
    val_ds = DecisionDataset(val_records, builder, a.max_length, shuffle_options=False) if val_records else None
    for name, ds in (("train", train_ds), ("val", val_ds)):
        if ds is not None and ds.too_long:
            say(f"{name}: {len(ds.too_long)} decisions longer than --max-length {a.max_length} skipped")
    say(f"{len(train_ds)} train / {len(val_ds) if val_ds else 0} val decisions; "
        f"prompt '{builder.chat_format}', options shuffled: {train_ds.shuffle}")

    # ---- model ----------------------------------------------------------------------------- #
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        a.dtype, torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, trust_remote_code=a.trust_remote_code)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = a.lora_targets if a.lora_targets == "all-linear" else [t.strip() for t in a.lora_targets.split(",")]
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    if rank0:
        model.print_trainable_parameters()

    collator = DecisionCollator(tok.pad_token_id)
    os.makedirs(a.out, exist_ok=True)
    if val_ds is not None and rank0:
        before = evaluate(model, val_ds, collator, a.eval_batch_size)
        say(f"val before training: {json.dumps(before['all'])}")

    import inspect

    # transformers 5 folded warmup_ratio into warmup_steps (a float in [0, 1) is a ratio).
    warmup = ({"warmup_ratio": a.warmup_ratio} if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
              else {"warmup_steps": a.warmup_ratio})
    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        **warmup, weight_decay=a.weight_decay, lr_scheduler_type="cosine",
        logging_steps=a.logging_steps, save_strategy="steps" if a.save_steps else "epoch",
        save_steps=a.save_steps or 500, save_total_limit=2, bf16=dtype == torch.bfloat16,
        remove_unused_columns=False, report_to=a.report_to, seed=a.seed,
        ddp_find_unused_parameters=False, dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available())
    trainer = build_trainer(model, train_ds, collator, targs, eval_ds=val_ds, eval_batch_size=a.eval_batch_size,
                            best_dir=os.path.join(a.out, "best"))
    trainer.train()

    if not trainer.is_world_process_zero():
        return 0
    final_dir = os.path.join(a.out, "final")
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    best_dir = os.path.join(a.out, "best")
    if not os.path.exists(best_dir):                     # no validation set: the final adapter is the result
        model.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    report = {"meta": metadata(a.model, vars(a), len(train_ds), len(val_ds) if val_ds else 0)}
    if val_ds is not None:
        report["val_before"] = {k: v for k, v in before.items() if k != "items"}
        after = evaluate(model, val_ds, collator, a.eval_batch_size)
        report["val_final"] = {k: v for k, v in after.items() if k != "items"}
        say(f"val after training:  {json.dumps(after['all'])}")
    for d in (final_dir, best_dir):
        with open(os.path.join(d, "salsa_jev.json"), "w") as fh:
            json.dump(report["meta"], fh, indent=2)
    with open(os.path.join(a.out, "train_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    if a.merge_to:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, trust_remote_code=a.trust_remote_code)
        merged = PeftModel.from_pretrained(base, best_dir).merge_and_unload()
        merged.save_pretrained(a.merge_to)
        tok.save_pretrained(a.merge_to)
        with open(os.path.join(a.merge_to, "salsa_jev.json"), "w") as fh:
            json.dump(report["meta"], fh, indent=2)
        say(f"merged model -> {a.merge_to}")
    say(f"adapter -> {best_dir}; report -> {os.path.join(a.out, 'train_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
