"""SALSA fine-tuning for Jev-style typed decisions on your own data.

Data -- JSONL, one decision per line, in JevBench's record format (JevBench's own
files load as they are):

  {"id": "t-001",                                   # optional, used for splits and reports
   "state": "text" | {...},                         # what the decision is about
   "question": {"type": "noul" | "choice" | "score",
                "instructions": "the decision rule",
                "criteria": ...},                   # as in the /v1/systemone API (docs/DECISION_API.md)
   "expected": "refund" | "yes" | 2,                # gold option key / yes|no / score level
   "probs": {"refund": 0.8, "cancel": 0.2},         # optional soft target; overrides `expected`
   "weight": 1.0,                                   # optional per-item loss weight
   "group": "g-17",                                 # optional: paraphrases share a group (kept together in splits)
   "family": "routing"}                             # optional: reported per family

What is trained is what is served: every item is rendered with the serving prompt
(``src/jev/decision.PromptBuilder``), and the loss is SALSA's single-token objective --
cross-entropy over *this item's* class tokens at the answer position (2-26 options,
or 2-10 score levels), against the gold option or the soft target.  The model is not
asked to generate anything else.

Choice items are re-rendered with their options in a random order every time they are
drawn (``shuffle_options``): the letter of the right answer moves, so the adapter learns
the option, not the position.  Off-the-shelf readers change 7-15% of their choice
answers when the options are reversed (docs/JEVBENCH_RESULTS.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass

import torch

from .decision import (
    PROMPT_VERSION,
    BadQuestion,
    PromptBuilder,
    TypedQuestion,
    UnsupportedQuestion,
    parse_question,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DataError(ValueError):
    pass


def _target(rec: dict, keys: tuple) -> list[float]:
    """Target distribution over ``keys`` from ``probs`` (soft) or ``expected`` (hard)."""
    probs = rec.get("probs")
    if probs is not None:
        if set(map(str, probs)) != set(keys):
            raise DataError(f"probs keys {sorted(probs)} != answer keys {sorted(keys)}")
        p = [float(probs[k]) for k in keys]
        total = sum(p)
        if total <= 0 or any(v < 0 for v in p):
            raise DataError("probs must be non-negative with a positive sum")
        return [v / total for v in p]
    exp = rec.get("expected")
    if isinstance(exp, bool):
        exp = "yes" if exp else "no"
    exp = str(exp) if exp is not None else None
    if exp not in keys:
        raise DataError(f"expected {rec.get('expected')!r} is not one of {list(keys)}")
    return [1.0 if k == exp else 0.0 for k in keys]


def load_records(path: str) -> tuple[list[dict], list[str]]:
    """Read and validate a decision JSONL file.  Returns (records, problems)."""
    records, problems = [], []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "state" not in rec or "question" not in rec:
                    raise DataError("needs 'state' and 'question'")
                q = parse_question(rec["question"])
                if rec.get("labels") is not None and set(map(str, rec["labels"])) != set(q.keys):
                    raise DataError(f"labels {rec['labels']} do not match the criteria keys {list(q.keys)}")
                _target(rec, q.keys)
                rec.setdefault("id", f"{os.path.basename(path)}:{n}")
                records.append(rec)
            except (json.JSONDecodeError, DataError, BadQuestion, UnsupportedQuestion) as e:
                problems.append(f"{path}:{n}: {type(e).__name__}: {e}")
    return records, problems


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _fingerprint(rec: dict) -> str:
    state = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((_norm(state) + "\x00" + _norm(rec["question"]["instructions"])).encode()).hexdigest()


def drop_benchmark_overlap(records: list[dict], benchmark_files: list[str]) -> tuple[list[dict], list[str]]:
    """Remove items whose normalised state + instructions equal a benchmark item's.

    The check JevBench itself runs on submitted training data: an exact match after
    lower-casing and collapsing whitespace.  Training on benchmark items makes any later
    score on that benchmark meaningless.
    """
    seen = set()
    for path in benchmark_files:
        with open(path, encoding="utf-8") as fh:
            seen.update(_fingerprint(json.loads(line)) for line in fh if line.strip())
    kept, dropped = [], []
    for rec in records:
        (dropped.append(rec["id"]) if _fingerprint(rec) in seen else kept.append(rec))
    return kept, dropped


def split_records(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic split by `group` (or id), so paraphrases never straddle train and val."""
    def bucket(rec):
        key = str(rec.get("group") or rec["id"])
        return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    train = [r for r in records if bucket(r) >= val_fraction]
    val = [r for r in records if bucket(r) < val_fraction]
    return train, val


def _with_order(question: dict, keys: list[str]) -> dict:
    q = dict(question)
    q["criteria"] = {k: question["criteria"][k] for k in keys}
    return q


@dataclass
class Encoded:
    input_ids: list[int]
    class_ids: list[int]
    target: list[float]
    weight: float


class DecisionDataset(torch.utils.data.Dataset):
    """Renders each decision with the serving prompt; choice options optionally shuffled."""

    def __init__(self, records, builder: PromptBuilder, max_length: int, shuffle_options: bool, seed: int = 0):
        self.builder, self.shuffle, self.rng = builder, shuffle_options, random.Random(seed)
        self.records, self.too_long = [], []
        for rec in records:
            ids, _ = builder.encode(rec["state"], parse_question(rec["question"]))
            if len(ids) + 1 > max_length:
                self.too_long.append(rec["id"])     # never truncated: a cut state is a different decision
            else:
                self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def encode(self, rec: dict, shuffle: bool) -> Encoded:
        question = rec["question"]
        if shuffle and question["type"] == "choice":
            keys = list(question["criteria"])
            self.rng.shuffle(keys)
            question = _with_order(question, keys)
        q: TypedQuestion = parse_question(question)
        ids, class_ids = self.builder.encode(rec["state"], q)
        return Encoded(ids, class_ids, _target(rec, q.keys), float(rec.get("weight", 1.0)))

    def __getitem__(self, i):
        e = self.encode(self.records[i], self.shuffle)
        return {"input_ids": e.input_ids, "class_ids": e.class_ids, "target": e.target, "weight": e.weight}


class DecisionCollator:
    """Right-pads prompts and option sets; records each row's answer position."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        T = max(len(b["input_ids"]) for b in batch)
        K = max(len(b["class_ids"]) for b in batch)
        input_ids = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), T), dtype=torch.long)
        class_ids = torch.zeros((len(batch), K), dtype=torch.long)
        class_mask = torch.zeros((len(batch), K), dtype=torch.bool)
        target = torch.zeros((len(batch), K), dtype=torch.float32)
        for r, b in enumerate(batch):
            n, k = len(b["input_ids"]), len(b["class_ids"])
            input_ids[r, :n] = torch.tensor(b["input_ids"])
            attention[r, :n] = 1
            class_ids[r, :k] = torch.tensor(b["class_ids"])
            class_mask[r, :k] = True
            target[r, :k] = torch.tensor(b["target"])
        return {"input_ids": input_ids, "attention_mask": attention, "answer_pos": attention.sum(1) - 1,
                "class_ids": class_ids, "class_mask": class_mask, "target": target,
                "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32)}


# --------------------------------------------------------------------------- #
# Readout and loss
# --------------------------------------------------------------------------- #
def class_logprobs(model, batch) -> torch.Tensor:
    """log-softmax over each row's class tokens at its answer position: [B, K] (padding = -inf).

    Only the answer positions go through the output head (``logits_to_keep`` with the
    positions in the batch), never a [B, T, vocabulary] tensor.
    """
    pos = batch["answer_pos"]
    uniq, inverse = torch.unique(pos, return_inverse=True)
    kwargs = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    try:
        logits = model(**kwargs, logits_to_keep=uniq).logits                     # [B, len(uniq), V]
        last = logits[torch.arange(len(pos), device=logits.device), inverse]
    except TypeError:                                                           # model without logits_to_keep
        logits = model(**kwargs).logits
        last = logits[torch.arange(len(pos), device=logits.device), pos]
    cls = last.float().gather(1, batch["class_ids"]).masked_fill(~batch["class_mask"], float("-inf"))
    return torch.log_softmax(cls, dim=-1)


def decision_loss(logp: torch.Tensor, batch) -> torch.Tensor:
    per_item = -(batch["target"] * logp.masked_fill(~batch["class_mask"], 0.0)).sum(-1)
    w = batch["weight"].to(per_item.device)
    return (per_item * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model, dataset: DecisionDataset, collator, batch_size: int = 8, device=None) -> dict:
    """Accuracy, NLL, Brier and ECE (10 equal-width bins) in canonical option order."""
    model.eval()
    device = device or next(model.parameters()).device
    rows = []
    for start in range(0, len(dataset), batch_size):
        recs = dataset.records[start:start + batch_size]
        enc = [dataset.encode(r, shuffle=False) for r in recs]
        batch = collator([{"input_ids": e.input_ids, "class_ids": e.class_ids, "target": e.target,
                           "weight": e.weight} for e in enc])
        batch = {k: v.to(device) for k, v in batch.items()}
        p = class_logprobs(model, batch).exp().cpu()
        for r, rec in enumerate(recs):
            k = len(enc[r].class_ids)
            probs, target = p[r, :k].tolist(), enc[r].target
            gold = max(range(k), key=target.__getitem__)
            pred = max(range(k), key=probs.__getitem__)
            keys = parse_question(rec["question"]).keys
            rows.append({"id": rec["id"], "type": rec["question"]["type"], "family": rec.get("family"),
                         "probs": dict(zip(keys, probs)), "correct": pred == gold,
                         "nll": -math.log(max(probs[gold], 1e-12)),
                         "brier": sum((a - b) ** 2 for a, b in zip(probs, target)), "conf": max(probs)})
    model.train()
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    def block(rs):
        if not rs:
            return None
        bins = [[] for _ in range(10)]
        for r in rs:
            bins[min(int(r["conf"] * 10), 9)].append(r)
        ece = sum(len(b) / len(rs) * abs(sum(x["correct"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b))
                  for b in bins if b)
        return {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
                "nll": sum(r["nll"] for r in rs) / len(rs), "brier": sum(r["brier"] for r in rs) / len(rs),
                "ece": ece}
    out = {"all": block(rows), "by_type": {t: block([r for r in rows if r["type"] == t])
                                          for t in sorted({r["type"] for r in rows})}}
    fams = sorted({r["family"] for r in rows if r["family"]})
    if fams:
        out["by_family"] = {f: block([r for r in rows if r["family"] == f]) for f in fams}
    out["items"] = rows
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_trainer(model, train_ds, collator, training_args, eval_ds=None, eval_batch_size=8, best_dir=None):
    from transformers import Trainer, TrainerCallback

    class DecisionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logp = class_logprobs(model, inputs)
            loss = decision_loss(logp, inputs)
            return (loss, {"logp": logp}) if return_outputs else loss

    class DecisionEval(TrainerCallback):
        """Evaluates on the validation decisions after every epoch; keeps the best adapter by NLL."""

        def __init__(self):
            self.best = math.inf
            self.trainer = None

        def on_epoch_end(self, args, state, control, model=None, **kw):
            if eval_ds is None or not state.is_world_process_zero:
                return
            m = evaluate(model, eval_ds, collator, eval_batch_size)["all"]
            self.trainer.log({f"val_{k}": v for k, v in m.items()})
            if best_dir and m["nll"] < self.best:
                self.best = m["nll"]
                model.save_pretrained(best_dir)

    cb = DecisionEval()
    trainer = DecisionTrainer(model=model, args=training_args, train_dataset=train_ds,
                              data_collator=collator, callbacks=[cb])
    cb.trainer = trainer
    return trainer


def metadata(base_model: str, args: dict, n_train: int, n_val: int) -> dict:
    return {"format": "salsa-jev-adapter/1", "base_model": base_model, "prompt_version": PROMPT_VERSION,
            "readout": "softmax over single class tokens at the answer anchor",
            "n_train": n_train, "n_val": n_val, "args": args}
