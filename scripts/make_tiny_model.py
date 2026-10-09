"""Write a miniature, randomly initialised copy of a model for CPU dry runs.

Keeps the real architecture class, the real tokenizer and chat template, and the
real vocabulary, and shrinks only depth and width -- so a dry run exercises the same
code paths as the real run (for Qwen3.5/3.8: the Gated-DeltaNet linear-attention
layers *and* the gated full-attention layers, the PEFT target-module names, the
chat template) at a size that runs on CPU in seconds.

    python scripts/make_tiny_model.py --model Qwen/Qwen3.8-27B --out /tmp/tiny-qwen38

The weights are random: anything trained from this is meaningless as a model and
only proves the plumbing.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def shrink(cfg, n_layers: int = 4):
    """Shrink a (possibly nested) causal-LM config in place and return it."""
    tc = getattr(cfg, "text_config", None) or cfg
    tc.num_hidden_layers = n_layers
    tc.hidden_size = 128
    tc.intermediate_size = 256
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    if getattr(tc, "head_dim", None):
        tc.head_dim = 32
    layer_types = getattr(tc, "layer_types", None)
    if layer_types and "linear_attention" not in layer_types:
        # Plain models also list per-layer attention kinds (full / sliding): keep them, shortened.
        tc.layer_types = list(layer_types[:n_layers])
    # Hybrid linear-attention models (Qwen3-Next / Qwen3.5 / Qwen3.8).
    elif layer_types:
        interval = int(getattr(tc, "full_attention_interval", 4) or 4)
        tc.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)
        ]
        for k, v in (
            ("linear_num_key_heads", 2),
            ("linear_num_value_heads", 4),
            ("linear_key_head_dim", 16),
            ("linear_value_head_dim", 16),
        ):
            if hasattr(tc, k):
                setattr(tc, k, v)
    if hasattr(tc, "mtp_num_hidden_layers"):
        tc.mtp_num_hidden_layers = 0
    return cfg


def build(model_id: str, cache_dir: str | None, n_layers: int = 4, seed: int = 0, trust_remote_code: bool = False):
    cfg = shrink(AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code), n_layers)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code).to(torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", default="cache_model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--trust-remote-code", action="store_true", help="only for models that need custom code")
    args = ap.parse_args()

    model, tok = build(args.model, args.cache_dir, args.layers, trust_remote_code=args.trust_remote_code)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    kinds = getattr(getattr(model.config, "text_config", model.config), "layer_types", None)
    n = sum(p.numel() for p in model.parameters())
    print(f"[tiny] {type(model).__name__} {n/1e6:.1f}M params, layers={kinds or args.layers} -> {args.out}")


if __name__ == "__main__":
    main()
