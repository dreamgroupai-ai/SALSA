"""Utilities for loading PEFT LoRA adapter checkpoints with key remapping."""

import json
import os
import shutil
import tempfile
from typing import Tuple


def fix_adapter_key_prefix(checkpoint_path: str) -> Tuple[str, bool]:
    """Fix ``language_model`` key prefix mismatch in saved LoRA adapter weights.

    Qwen3.5 training saves adapter keys as
    ``base_model.model.model.language_model.layers.*`` but
    ``AutoModelForCausalLM`` exposes layers under ``model.layers.*`` (no
    ``language_model`` wrapper).  PEFT cannot match any keys and silently
    drops all LoRA weights.

    This function detects the mismatch, creates a temporary directory with
    remapped adapter files, and returns the path to use for
    ``PeftModel.from_pretrained``.

    Args:
        checkpoint_path: Path to the PEFT checkpoint directory containing
            ``adapter_model.safetensors`` and ``adapter_config.json``.

    Returns:
        ``(path_to_use, needs_cleanup)`` -- *path_to_use* is either the
        original *checkpoint_path* (no mismatch) or a temporary directory
        with fixed keys.  The caller must ``shutil.rmtree(path_to_use)``
        when *needs_cleanup* is ``True``.
    """
    from safetensors.torch import load_file, save_file

    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        return checkpoint_path, False

    tensors = load_file(adapter_path)
    needs_fix = any(".language_model." in k for k in tensors.keys())
    if not needs_fix:
        del tensors
        return checkpoint_path, False

    print(
        "Detected language_model prefix mismatch in adapter weights -- "
        "remapping keys for PeftModel.from_pretrained ..."
    )

    tmp_dir = tempfile.mkdtemp(prefix="salsa_adapter_fix_")
    remapped = {k.replace(".language_model.", ".", 1): v for k, v in tensors.items()}
    save_file(remapped, os.path.join(tmp_dir, "adapter_model.safetensors"))
    del tensors, remapped

    config_src = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(tmp_dir, "adapter_config.json"))

    return tmp_dir, True


def get_adapter_lora_rank(checkpoint_path: str, default: int = 16) -> int:
    """Read the LoRA rank from an adapter's config file."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f).get("r", default)
    return default
