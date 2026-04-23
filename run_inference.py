#!/usr/bin/env python
"""
Inference script for running predictions on test set using a fine-tuned checkpoint.

Usage:
    # Multi-GPU with accelerate
    accelerate launch run_inference.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint

    # Single GPU
    python run_inference.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint

    # With custom output path
    python run_inference.py --config-name=task13a \
        model.load_checkpoint=/path/to/checkpoint \
        output.submission_path=my_submission.csv
"""

import os
import shutil
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator

from src.adapter_utils import fix_adapter_key_prefix
from src.manage_datasets import SalsaDatasetFactory


def collate_fn_factory(tokenizer):
    """Create a collate function for batching inference samples."""
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        sample_ids = [item.get("sample_id", -1) for item in batch]
        
        # Pad sequences (left padding for causal LM inference)
        max_len = max(len(ids) for ids in input_ids)
        pad_token_id = tokenizer.pad_token_id or 0
        
        padded_input_ids = torch.stack([
            F.pad(ids, (max_len - len(ids), 0), value=pad_token_id)
            for ids in input_ids
        ])
        padded_attention_mask = torch.stack([
            F.pad(mask, (max_len - len(mask), 0), value=0)
            for mask in attention_mask
        ])
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        }
    return collate_fn


def run_inference(model, dataloader, target_token_ids, accelerator):
    """
    Run inference on the test set.
    
    Args:
        model: The language model
        dataloader: DataLoader for test samples
        target_token_ids: List of token IDs for classification labels
        accelerator: Accelerator instance
        
    Returns:
        List of (sample_id, prediction) tuples
    """
    model.eval()
    all_predictions = []
    all_sample_ids = []
    
    target_ids = torch.tensor(target_token_ids, device=accelerator.device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference", disable=not accelerator.is_main_process):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            sample_ids = batch["sample_ids"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]
            
            # Get logits at the last non-padded position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_indices, seq_lengths, :]  # [B, V]
            
            # Extract logits for target tokens only
            target_logits = last_logits.index_select(dim=-1, index=target_ids)  # [B, K]
            
            # Get predicted class (0 or 1)
            preds = target_logits.argmax(dim=-1)
            
            all_predictions.append(preds)
            all_sample_ids.append(sample_ids)
    
    # Concatenate all predictions and sample_ids
    all_predictions = torch.cat(all_predictions, dim=0)
    all_sample_ids_tensor = torch.cat(all_sample_ids, dim=0)
    
    # Gather predictions and sample_ids from all processes
    all_predictions = accelerator.gather(all_predictions)
    all_sample_ids_tensor = accelerator.gather(all_sample_ids_tensor)
    
    return all_predictions.cpu().tolist(), all_sample_ids_tensor.cpu().tolist()


def save_submission(predictions, sample_ids, filepath):
    """Save predictions in submission format (ID,label)."""
    df = pd.DataFrame({
        "ID": sample_ids,
        "label": predictions
    })
    df.to_csv(filepath, index=False)
    print(f"Saved submission to {filepath}")


@hydra.main(config_path="configs", config_name="task13a", version_base="1.3")
def main(config: DictConfig):
    # Initialize Accelerator with bf16 mixed precision (matching training)
    accelerator = Accelerator(mixed_precision='bf16')
    
    if accelerator.is_main_process:
        print("=" * 60)
        print("SALSA Inference Script")
        print("=" * 60)
        print(f"Config: {OmegaConf.to_yaml(config, resolve=True)}")
    
    # Get checkpoint path from config
    checkpoint_path = getattr(config.model, 'load_checkpoint', None)
    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint path provided. Use: "
            "model.load_checkpoint=/path/to/checkpoint"
        )
    
    if accelerator.is_main_process:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Get HuggingFace token
    hugging_token = os.environ.get("HuggingfaceToken")
    if not hugging_token:
        print("Warning: HuggingfaceToken not set, using cached model")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        token=hugging_token,
        cache_dir=config.model.cache_dir,
    )
    tokenizer.truncation_side = "left"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    if accelerator.is_main_process:
        print(f"\nLoading base model: {config.model.id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        token=hugging_token,
        cache_dir=config.model.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,  # Let accelerate handle device placement
    )
    
    # Load LoRA adapter
    if accelerator.is_main_process:
        print(f"Loading LoRA adapter from: {checkpoint_path}")
    
    fixed_path, needs_adapter_cleanup = fix_adapter_key_prefix(checkpoint_path)
    model = PeftModel.from_pretrained(model, fixed_path)
    if needs_adapter_cleanup:
        shutil.rmtree(fixed_path, ignore_errors=True)
    
    # Optionally merge LoRA weights for faster inference
    if getattr(config.model, 'merge_loaded_lora', True):
        if accelerator.is_main_process:
            print("Merging LoRA weights...")
        model = model.merge_and_unload()
    
    model.eval()
    
    # Prepare dataset using SalsaDatasetFactory
    if accelerator.is_main_process:
        print(f"\nPreparing test dataset...")
    
    max_seq_length = getattr(config.training, "max_seq_length", 8192)
    factory = SalsaDatasetFactory(config, max_seq_length=max_seq_length)
    
    # Get target token IDs for classification
    target_token_ids = factory.target_token_ids
    if accelerator.is_main_process:
        print(f"Target token IDs: {target_token_ids}")
        print(f"Label options: {config.classifier.options['label']}")
    
    # Load test dataset with IDs
    test_dataset, test_ids = factory.compile_dataset_with_ids(config.dataset.test_name)
    
    if accelerator.is_main_process:
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    batch_size = getattr(config.training, 'per_device_eval_batch_size', 2)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_factory(tokenizer),
        num_workers=0,
    )
    
    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    if accelerator.is_main_process:
        print(f"\nStarting inference...")
        print(f"Batch size per device: {batch_size}")
        print(f"Number of processes: {accelerator.num_processes}")
    
    # Run inference
    predictions, sample_ids = run_inference(model, dataloader, target_token_ids, accelerator)
    
    # Save results on main process only
    if accelerator.is_main_process:
        # Deduplicate gathered results (DistributedSampler may pad with duplicates)
        # Check if dataset actually embedded sample_ids (not all -1)
        has_real_ids = any(sid != -1 for sid in sample_ids)
        if has_real_ids:
            seen = {}
            for sid, pred in zip(sample_ids, predictions):
                if sid not in seen and sid != -1:
                    seen[sid] = pred
            final_ids = list(seen.keys())
            final_predictions = list(seen.values())
        elif test_ids is not None:
            final_ids = test_ids
            final_predictions = predictions[:len(final_ids)]
        else:
            final_ids = list(range(len(test_dataset)))
            final_predictions = predictions[:len(final_ids)]
        
        # Determine output path
        output_path = getattr(config.output, 'submission_path', None)
        if output_path is None:
            # Default: save next to checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                os.path.dirname(checkpoint_path),
                f"submission_{timestamp}.csv"
            )
        
        save_submission(final_predictions, final_ids, output_path)
        
        print("\n" + "=" * 60)
        print("Inference complete!")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Output saved to: {output_path}")
        print("=" * 60)
    
    # Wait for all processes
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
