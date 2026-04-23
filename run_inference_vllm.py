#!/usr/bin/env python
"""
vLLM-based inference script for high-throughput test set predictions.

Uses single-token generation with logprobs to classify samples efficiently.
The prompt template is aligned with training via SalsaDatasetFactory.

Supports both LoRA-adapter inference and base-model-only (zero-shot) inference.
When no checkpoint is provided (or load_checkpoint is None), the script runs
the base model directly without any adapter.

Usage:
    # Zero-shot inference (base model only, no adapter)
    python run_inference_vllm.py --config-name=task13a32B \
        +output.submission_path=submission_zeroshot.csv

    # With LoRA adapter checkpoint
    python run_inference_vllm.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint

    # With explicit tensor parallelism
    python run_inference_vllm.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint \
        inference.tensor_parallel_size=4

    # With custom batch size
    python run_inference_vllm.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint \
        inference.batch_size=64
"""

import json
import os
import warnings
from datetime import datetime
from typing import List, Optional

import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.manage_datasets import SalsaDatasetFactory


def _qwen35_text_config_override(config):
    """Unwrap Qwen3.5 VL config to text-only before the model runner initialises.

    The HF Hub config for Qwen3.5 is a VL wrapper with nested ``text_config``.
    If left as-is, vLLM treats the model as multimodal and creates 3D M-RoPE
    position tensors that are incompatible with the text-only rotary embedding,
    causing ``RuntimeError: query, key and positions must have the same
    batch_size and seq_len``.  Applying this via ``hf_overrides`` ensures the
    config is patched *before* the model runner reads it.
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "linear_num_key_heads"):
        if not hasattr(text_cfg, "architectures") or not text_cfg.architectures:
            text_cfg.architectures = getattr(
                config, "architectures", ["Qwen3_5ForCausalLM"]
            )
        rope_params = getattr(text_cfg, "rope_parameters", None)
        if rope_params and isinstance(rope_params, dict):
            rope_params.pop("mrope_section", None)
            rope_params.pop("mrope_interleaved", None)
        return text_cfg
    return config


def prepare_lora_adapter(checkpoint_path: str) -> tuple:
    """Prepare a PEFT LoRA adapter for use with vLLM native LoRA.

    Delegates key remapping to ``fix_adapter_key_prefix`` in ``adapter_utils``
    and reads the LoRA rank from ``adapter_config.json``.

    Returns ``(adapter_path, lora_rank, needs_cleanup)``.
    """
    from src.adapter_utils import fix_adapter_key_prefix, get_adapter_lora_rank

    adapter_safetensors = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_safetensors):
        raise FileNotFoundError(
            f"adapter_model.safetensors not found in {checkpoint_path}"
        )

    lora_rank = get_adapter_lora_rank(checkpoint_path)
    fixed_path, needs_cleanup = fix_adapter_key_prefix(checkpoint_path)
    return fixed_path, lora_rank, needs_cleanup


def prepare_prompts(config: DictConfig) -> tuple:
    """
    Prepare prompts for inference using SalsaDatasetFactory.
    
    Uses factory methods to ensure identical formatting as run_inference.py:
    - Chat template with enable_thinking=False (adds <think>\n\n</think>\n\n wrapper)
    - Answer prefix appended (e.g., <ANSWER>)
    - Left-side truncation to preserve label position
    
    Returns:
        Tuple of (prompts, sample_ids, target_token_ids)
    """
    max_seq_length = getattr(config.training, "max_seq_length", 8192)
    factory = SalsaDatasetFactory(config, max_seq_length=max_seq_length)
    
    # Get target token IDs from factory
    target_token_ids = factory.target_token_ids
    
    # Use factory's tokenizer which has truncation_side="left" set
    tokenizer = factory.tokenizer
    
    # Load raw dataset
    hf_dataset = factory._load_dataset(config.dataset.test_name)
    
    # Extract IDs if present
    sample_ids = None
    if "ID" in hf_dataset.column_names:
        sample_ids = hf_dataset["ID"]
    
    # Use factory's formatting methods (includes enable_thinking=False + answer_prefix)
    # This ensures identical prompt format as run_inference.py
    dataset = factory._compile_dataframe_inference(hf_dataset)
    formatted = factory._formatting_inference(dataset)
    
    # Get prompts and apply left-side truncation for vLLM
    prompts = []
    truncated_count = 0
    
    for text in formatted["text"]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_seq_length:
            truncated_count += 1
            # Left-side truncation: keep last max_seq_length tokens (preserves label position)
            token_ids = tokenizer(
                text, 
                truncation=True, 
                max_length=max_seq_length,
                add_special_tokens=False
            )["input_ids"]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
        prompts.append(text)
    
    if truncated_count > 0:
        print(f"Truncated {truncated_count} prompts that exceeded {max_seq_length} tokens")
    
    return prompts, sample_ids, target_token_ids


def run_vllm_inference(
    llm: LLM,
    prompts: List[str],
    target_token_ids: List[int],
    batch_size: int = 32,
    lora_request: Optional[LoRARequest] = None,
) -> List[int]:
    """
    Run inference using vLLM with single-token generation.
    
    Args:
        llm: vLLM LLM instance
        prompts: List of formatted prompts
        target_token_ids: Token IDs for classification labels (e.g., [token_id_for_0, token_id_for_1])
        batch_size: Batch size for processing
        lora_request: Optional LoRA adapter request for vLLM native LoRA
        
    Returns:
        List of predicted labels (0 or 1)
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(target_token_ids),
        allowed_token_ids=target_token_ids,
    )
    
    predictions = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="vLLM inference"):
        batch_prompts = prompts[i:i + batch_size]
        
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        
        for output in outputs:
            generated_token = output.outputs[0].token_ids[0] if output.outputs[0].token_ids else None
            if generated_token in target_token_ids:
                pred = target_token_ids.index(generated_token)
            else:
                pred = 0
            
            predictions.append(pred)
    
    return predictions


def save_submission(predictions: List[int], sample_ids: Optional[List], filepath: str):
    """Save predictions in submission format (ID,label)."""
    if sample_ids is None:
        sample_ids = list(range(len(predictions)))
    
    df = pd.DataFrame({
        "ID": sample_ids,
        "label": predictions
    })
    df.to_csv(filepath, index=False)
    print(f"Saved submission to {filepath}")


@hydra.main(config_path="configs", config_name="task13a", version_base="1.3")
def main(config: DictConfig):
    print("=" * 60)
    print("SALSA vLLM Inference Script")
    print("=" * 60)

    checkpoint_path = getattr(config.model, 'load_checkpoint', None)
    if checkpoint_path in (None, "None", "null", ""):
        checkpoint_path = None

    use_lora = checkpoint_path is not None

    if use_lora:
        print(f"Mode: LoRA adapter inference")
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print(f"Mode: Base model only (zero-shot, no adapter)")
    print(f"Base model: {config.model.id}")

    hugging_token = os.environ.get("HuggingfaceToken")
    if not hugging_token:
        print("Warning: HuggingfaceToken not set, using cached model")

    # Inference settings (avoid torch.cuda.device_count() which initializes CUDA
    # and forces vLLM to use 'spawn' multiprocessing instead of 'fork')
    inference_config = getattr(config, 'inference', None) or {}
    tensor_parallel_size = getattr(inference_config, 'tensor_parallel_size', None)
    if tensor_parallel_size is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        tensor_parallel_size = len(visible.split(",")) if visible else 8
    batch_size = getattr(inference_config, 'batch_size', 32)
    gpu_memory_utilization = getattr(inference_config, 'gpu_memory_utilization', 0.9)

    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Batch size: {batch_size}")

    # Resolve cache_dir to absolute path (Hydra changes cwd)
    cache_dir = os.path.abspath(
        os.path.join(hydra.utils.get_original_cwd(), config.model.cache_dir)
    )

    # Conditionally prepare LoRA adapter
    lora_request = None
    adapter_needs_cleanup = False
    lora_rank = None

    if use_lora:
        print("\n" + "-" * 40)
        print("Preparing LoRA adapter for native vLLM inference...")
        print("-" * 40)
        adapter_path, lora_rank, adapter_needs_cleanup = prepare_lora_adapter(
            checkpoint_path
        )
        lora_request = LoRARequest("adapter", 1, adapter_path)
        print(f"Adapter path: {adapter_path}")
        print(f"LoRA rank: {lora_rank}")

    # Initialize vLLM BEFORE prepare_prompts() so CUDA is not yet initialized
    # and vLLM can use 'fork' multiprocessing (avoids PTX compat issues with 'spawn')
    print("\n" + "-" * 40)
    if use_lora:
        print("Initializing vLLM with native LoRA...")
    else:
        print("Initializing vLLM with base model (no adapter)...")
    print("-" * 40)

    warnings.filterwarnings(
        "ignore",
        message=r"Unrecognized keys in `rope_parameters`.*mrope",
    )

    llm_kwargs = dict(
        model=config.model.id,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=getattr(config.training, "max_seq_length", 8192) + 1024,
        hf_token=hugging_token,
        hf_overrides=_qwen35_text_config_override,
        enforce_eager=True,
    )

    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank

    llm = LLM(**llm_kwargs)

    # Prepare prompts after vLLM is initialized
    print("\nPreparing prompts...")
    prompts, sample_ids, target_token_ids = prepare_prompts(config)
    print(f"Total prompts: {len(prompts)}")
    print(f"Target token IDs: {target_token_ids}")
    print(f"Label options: {config.classifier.options['label']}")

    # Run inference
    print("\n" + "-" * 40)
    print("Running inference...")
    print("-" * 40)

    predictions = run_vllm_inference(
        llm=llm,
        prompts=prompts,
        target_token_ids=target_token_ids,
        batch_size=batch_size,
        lora_request=lora_request,
    )

    # Determine output path
    output_path = getattr(config.output, 'submission_path', None)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if checkpoint_path:
            output_path = os.path.join(
                os.path.dirname(checkpoint_path),
                f"submission_vllm_{timestamp}.csv"
            )
        else:
            original_cwd = hydra.utils.get_original_cwd()
            output_path = os.path.join(
                original_cwd,
                f"submission_vllm_zeroshot_{timestamp}.csv"
            )

    # Save results
    save_submission(predictions, sample_ids, output_path)

    # Clean up temp adapter directory if we created one
    if adapter_needs_cleanup:
        import shutil
        shutil.rmtree(adapter_path, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
