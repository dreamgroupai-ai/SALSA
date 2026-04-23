"""
End-to-end training script:
- Custom dataset factory (chat template + single-token label)
- SALSATrainer with custom loss over K label tokens
- Metrics: token-level accuracy + ROC-AUC (initial + per-epoch)
- Weights & Biases logging
- Accelerate-compatible (use `accelerate launch`)

Notes:
- Ensure env var HuggingfaceToken is set for gated models.
- Create accelerate_config.yaml and launch with Accelerate for multi-GPU.
"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from sklearn.metrics import roc_auc_score, f1_score
import hydra
from omegaconf import DictConfig
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from collections import defaultdict
import random

from src.adapter_utils import fix_adapter_key_prefix
from src.manage_datasets import SalsaDatasetFactory
from accelerate import Accelerator, InitProcessGroupKwargs


# -------------------------
# Balanced Sampler
# -------------------------
class BalancedSampler(Sampler):
    """
    A sampler that balances samples across (label, language) combinations.

    Each epoch samples equally from each group, with a maximum of `max_repetitions`
    for any single sample per epoch to avoid overfitting to rare combinations.
    """

    def __init__(self, dataset, balance_columns, max_repetitions=3, seed=42):
        """
        Args:
            dataset: HuggingFace Dataset with balance columns preserved
            balance_columns: List of column names to balance on (e.g., ["original_label", "language"])
            max_repetitions: Maximum times a sample can appear in one epoch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.balance_columns = balance_columns
        self.max_repetitions = max_repetitions
        self.seed = seed
        self.epoch = 0

        # Build index groups based on balance column combinations
        # Batch-read columns (O(1) Arrow ops) instead of per-row random access
        self.group_indices = defaultdict(list)
        column_data = {col: dataset[col] for col in balance_columns if col in dataset.column_names}
        for idx in range(len(dataset)):
            group_key = tuple(column_data[col][idx] if col in column_data else None for col in balance_columns)
            self.group_indices[group_key].append(idx)

        self.groups = list(self.group_indices.keys())
        self.num_groups = len(self.groups)

        # Find the smallest group size
        self.min_group_size = min(len(indices) for indices in self.group_indices.values())

        # Target samples per group per epoch (limited by max_repetitions)
        self.samples_per_group = min(
            self.min_group_size * max_repetitions,
            max(len(indices) for indices in self.group_indices.values())
        )

        # Total samples per epoch
        self._length = self.samples_per_group * self.num_groups

        print(f"[BalancedSampler] Initialized with {self.num_groups} groups:")
        for group_key, indices in self.group_indices.items():
            print(f"  {group_key}: {len(indices)} samples")
        print(f"[BalancedSampler] Samples per group per epoch: {self.samples_per_group}")
        print(f"[BalancedSampler] Total samples per epoch: {self._length}")

    def __iter__(self):
        # Create RNG for this epoch
        rng = random.Random(self.seed + self.epoch)

        all_indices = []

        for group_key in self.groups:
            group_indices = self.group_indices[group_key].copy()
            rng.shuffle(group_indices)

            # Sample with repetition limit
            sampled = []
            repetition_count = defaultdict(int)

            while len(sampled) < self.samples_per_group:
                for idx in group_indices:
                    if repetition_count[idx] < self.max_repetitions:
                        sampled.append(idx)
                        repetition_count[idx] += 1
                        if len(sampled) >= self.samples_per_group:
                            break

                # If we've exhausted all samples at max repetitions, break
                if all(repetition_count[idx] >= self.max_repetitions for idx in group_indices):
                    break

            all_indices.extend(sampled)

        # Shuffle the combined indices
        rng.shuffle(all_indices)

        return iter(all_indices)

    def __len__(self):
        return self._length

    def set_epoch(self, epoch):
        """Set the epoch for reproducible shuffling."""
        self.epoch = epoch


# -------------------------
# Metrics wiring
# -------------------------
def make_preprocess_logits_for_metrics(target_token_ids):
    """
    Returns a function (logits, labels) -> sliced logits of shape (B, T, K),
    where K = len(target_token_ids). Compute_metrics will do the shift.
    """
    target_token_ids = list(target_token_ids)

    def _preprocess(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        # logits: [B, T, V]
        idx = torch.as_tensor(target_token_ids, device=logits.device, dtype=torch.long)
        return logits.index_select(dim=-1, index=idx)  # -> (B, T, K)

    return _preprocess


def make_compute_metrics(target_token_ids):
    """
    HF-compatible compute_metrics(eval_pred):
      - shifts for next-token prediction
      - remaps label token-ids -> class indices 0..K-1
      - reports token_acc, roc_auc, and F1 (macro)
    """
    label2idx = {tid: i for i, tid in enumerate(target_token_ids)}
    K = len(target_token_ids)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds are the sliced logits (B, T, K) from preprocess

        # Ensure ndarray
        if "torch" in str(type(preds)):
            preds = preds.detach().cpu().numpy()
        if "torch" in str(type(labels)):
            labels = labels.detach().cpu().numpy()

        if preds.ndim != 3 or preds.shape[-1] != K:
            raise ValueError(f"Expected preds of shape (B, T, K={K}), got {preds.shape}")

        # Shift for next-token prediction
        shift_preds = preds[:, :-1, :]  # (B, T-1, K)
        shift_labels = labels[:, 1:]    # (B, T-1)

        mask = shift_labels != -100
        if not np.any(mask):
            return {"token_acc": 0.0, "roc_auc": float("nan"), "f1": float("nan")}

        # Remap label token ids -> class indices
        y_tok = shift_labels[mask]  # 1D
        remapped = np.full_like(y_tok, -1, dtype=np.int64)
        for uid in np.unique(y_tok):
            if uid in label2idx:
                remapped[y_tok == uid] = label2idx[uid]

        valid = remapped != -1
        if not np.any(valid):
            return {"token_acc": 0.0, "roc_auc": float("nan"), "f1": float("nan")}

        y_true = remapped[valid]            # (N,)
        y_scores = shift_preds[mask][valid] # (N, K)
        y_pred = y_scores.argmax(-1)

        token_acc = float((y_pred == y_true).mean())

        # ROC-AUC (binary or multiclass OVR)
        roc_auc = float("nan")
        try:
            if np.unique(y_true).size >= 2:
                if K == 2:
                    roc_auc = float(roc_auc_score(y_true, y_scores[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro"))
        except Exception:
            pass

        # F1 (macro). Undefined with a single class -> nan.
        f1 = float("nan")
        try:
            if np.unique(y_true).size >= 2:
                f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except Exception:
            pass

        return {"token_acc": token_acc, "roc_auc": roc_auc, "f1": f1}

    return compute_metrics



# -------------------------
# Custom Trainer with loss
# -------------------------
class SALSATrainer(SFTTrainer):
    def __init__(self, *args, target_token_ids, balanced_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_token_ids = target_token_ids
        self.balanced_sampler = balanced_sampler

    def get_train_dataloader(self):
        """Override to use balanced sampler if configured."""
        if self.balanced_sampler is None:
            return super().get_train_dataloader()

        # Set epoch for reproducible shuffling (called AFTER state is restored from checkpoint)
        epoch = self.state.epoch if (hasattr(self.state, 'epoch') and self.state.epoch is not None) else 0
        int_epoch = int(epoch)

        if hasattr(self.balanced_sampler, 'set_epoch'):
            self.balanced_sampler.set_epoch(int_epoch)

        # Log for resume visibility - helps verify data skipping is working
        global_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
        if global_step > 0:
            print(f"[SALSATrainer] Creating dataloader for resumed training: "
                  f"epoch={int_epoch} (actual: {epoch:.4f}), global_step={global_step}")
        else:
            print(f"[SALSATrainer] Creating dataloader for fresh training: epoch={int_epoch}")

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=self.balanced_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        inputs.pop("probs", None)
        inputs.pop("weight", None)

        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        if labels is None:
            loss = outputs.get("loss", None) if isinstance(outputs, dict) else None
            return (loss, outputs) if return_outputs else loss

        # Shift for CLM (avoid .contiguous() on full logits to save memory)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_logits.device)

        # Remap vocab ids -> class indices (0..K-1), others -> -100
        target_ids = torch.as_tensor(self.target_token_ids, device=shift_logits.device, dtype=torch.long)
        K = target_ids.numel()

        remap = torch.full((shift_logits.size(-1),), -100, dtype=torch.long, device=shift_logits.device)
        remap[target_ids] = torch.arange(K, dtype=torch.long, device=shift_logits.device)

        pad_mask = shift_labels.eq(-100)
        shift_labels_safe = shift_labels.clamp_min(0)
        mapped_labels = remap[shift_labels_safe]
        mapped_labels[pad_mask] = -100

        # Slice logits down to the K classes
        shift_logits_K = shift_logits.index_select(dim=-1, index=target_ids)

        nonpad_mask = shift_labels.ne(-100)
        keep_idx = nonpad_mask.nonzero(as_tuple=True)
        logits_kept = shift_logits_K[keep_idx]         # (N, K)
        mapped_labels_kept = mapped_labels[keep_idx]   # (N,)

        log_probs = F.log_softmax(logits_kept, dim=-1)
        loss = (-log_probs[torch.arange(log_probs.size(0), device=log_probs.device), mapped_labels_kept]).mean()

        return (loss, outputs) if return_outputs else loss


# -------------------------
# Test Inference Callback
# -------------------------
class TestInferenceCallback(TrainerCallback):
    """
    Callback that runs inference on test set at each checkpoint save
    and saves predictions in submission format (ID,label).
    If test_labels are provided, computes and logs F1 score to wandb.
    """
    def __init__(self, test_dataset, test_ids, target_token_ids, output_dir, tokenizer, batch_size=8, test_labels=None):
        self.test_dataset = test_dataset
        self.test_ids = test_ids  # List of IDs from test.parquet
        self.target_token_ids = target_token_ids
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.test_labels = test_labels  # Ground truth labels for F1 computation

    def on_save(self, args, state, control, model=None, **kwargs):
        """Run inference on test set when a checkpoint is saved.

        IMPORTANT: With DeepSpeed ZeRO-3, model parameters are sharded across
        ranks. ALL ranks must participate in forward passes so that AllGather
        operations can reconstruct the full parameters. Running inference on
        only rank 0 would deadlock because the other ranks are not participating
        in the required collective communications.
        """
        if model is None:
            return

        import torch.distributed as dist

        # ALL ranks run inference (required for ZeRO-3 parameter gathering)
        if state.is_world_process_zero:
            print(f"\n[TestInferenceCallback] Running test inference at step {state.global_step}...")
        predictions = self._run_inference(model)

        # Only rank 0 saves results and logs metrics
        if state.is_world_process_zero:
            # Save submission file
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            submission_path = os.path.join(checkpoint_dir, "submission.csv")
            self._save_submission(predictions, submission_path)
            print(f"[TestInferenceCallback] Saved submission to {submission_path}")

            # Compute and log F1 if test labels are available
            if self.test_labels is not None:
                try:
                    test_f1 = f1_score(self.test_labels, predictions, average="macro", zero_division=0)
                    print(f"[TestInferenceCallback] Test F1 (macro): {test_f1:.4f}")
                    # Log to wandb
                    wandb.log({"test/f1": test_f1}, step=state.global_step)
                except Exception as e:
                    print(f"[TestInferenceCallback] Failed to compute F1: {e}")

        # Sync after inference so all ranks proceed together
        if dist.is_initialized():
            dist.barrier()

    def _run_inference(self, model):
        """Generate predictions for test set.

        NOTE: Do NOT unwrap model.module here. With DeepSpeed ZeRO-3,
        parameters are sharded and the wrapper is needed to trigger
        AllGather operations during forward passes. All ranks must
        call this method together.
        """
        model.eval()
        device = next(model.parameters()).device
        predictions = []

        # Create a simple dataloader (left-padding for causal LM inference)
        def collate_fn(batch):
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
            max_len = max(len(ids) for ids in input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0
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
            }

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        target_ids = torch.tensor(self.target_token_ids, device=device)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Test inference"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, T, V]

                # Get logits at the last position for each sequence
                # Find the actual last token position (before padding)
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_indices = torch.arange(logits.size(0), device=device)
                last_logits = logits[batch_indices, seq_lengths, :]  # [B, V]

                # Extract logits for target tokens only
                target_logits = last_logits.index_select(dim=-1, index=target_ids)  # [B, K]

                # Get predicted class (0 or 1)
                preds = target_logits.argmax(dim=-1).cpu().tolist()
                predictions.extend(preds)

        model.train()
        return predictions

    def _save_submission(self, predictions, filepath):
        """Save predictions in submission format (ID,label)."""
        if self.test_ids is None:
            # If no IDs, use sequential indices
            ids = list(range(len(predictions)))
        else:
            ids = self.test_ids

        df = pd.DataFrame({
            "ID": ids,
            "label": predictions
        })
        df.to_csv(filepath, index=False)


# -------------------------
# Orchestration
# -------------------------
class TaskTrainer:
    def __init__(self, config):
        self.config = config
        # Get max_seq_length from config (default in training/default.yaml)
        self.max_seq_length = getattr(config.training, "max_seq_length", 1024)
        factory = SalsaDatasetFactory(config, max_seq_length=self.max_seq_length)
        self.train_dataset = factory.compile_dataset("train")
        self.validation_dataset = factory.compile_dataset(config.dataset.val_name)
        # Load test dataset with IDs for submission
        if config.dataset.test_name:
            self.test_dataset, self.test_ids = factory.compile_dataset_with_ids(config.dataset.test_name)
            # Extract test labels if available (for F1 computation)
            self.test_labels = factory.get_test_labels(config.dataset.test_name)
        else:
            self.test_dataset, self.test_ids = None, None
            self.test_labels = None
        self.hugging_token = os.environ.get("HuggingfaceToken")
        self.factory = factory

        # Create balanced sampler if enabled in config
        self.balanced_sampler = None
        balanced_config = getattr(config.dataset, "balanced_sampling", None)
        if balanced_config and getattr(balanced_config, "enabled", False):
            balance_columns = getattr(balanced_config, "balance_columns", ["original_label", "language"])
            max_repetitions = getattr(balanced_config, "max_repetitions", 3)
            seed = getattr(config.training, "seed", 42)
            self.balanced_sampler = BalancedSampler(
                dataset=self.train_dataset,
                balance_columns=balance_columns,
                max_repetitions=max_repetitions,
                seed=seed,
            )
            print(f"[TaskTrainer] Balanced sampling enabled with columns: {balance_columns}, max_repetitions: {max_repetitions}")

    def train(self):
        # --- NCCL timeout: set before any distributed init ---
        # With MoE models + ZeRO-3, AllGather operations can be slow;
        # the default 600s NCCL timeout may be too short for large MoE layers.
        os.environ.setdefault("NCCL_TIMEOUT", "7200")  # 2 hours (in seconds)

        # --- W&B init ---
        os.environ.setdefault("WANDB_PROJECT", getattr(self.config.logging, "wandb_project", "salsa"))
        os.environ.setdefault("WANDB_RUN_GROUP", getattr(self.config.training, "experiment", "default"))
        os.environ.setdefault("WANDB_WATCH", "false")  # avoid heavy autowatch
        run_name = getattr(self.config.training, "run_name", None) or f"{self.config.model.id.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # ======== ACCELERATOR INIT ===========
        grad_accum = self.config.training.gradient_accumulation_steps
        print(f"[TaskTrainer] Resolved gradient_accumulation_steps={grad_accum}")
        nccl_timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        accelerator = Accelerator(
            log_with="wandb",
            mixed_precision='bf16',
            gradient_accumulation_steps=grad_accum,
            kwargs_handlers=[nccl_timeout],
        )
        if torch.cuda.is_available():
            local_rank = accelerator.local_process_index
            torch.cuda.set_device(local_rank)

        accelerator.init_trackers(
            project_name=os.environ["WANDB_PROJECT"],
            config=dict(self.config),
            init_kwargs={"wandb": {
                "name": run_name,
            }},
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.id,
            token=self.hugging_token,
            cache_dir=self.config.model.cache_dir,
        )

        base_collator = DataCollatorForSeq2Seq(tokenizer)

        # Get balance columns to remove from batch
        balanced_config = getattr(self.config.dataset, "balanced_sampling", None)
        balance_columns = []
        if balanced_config:
            balance_columns = getattr(balanced_config, "balance_columns", ["original_label", "language"])

        # Columns the model actually needs for forward pass
        _MODEL_KEYS = {"input_ids", "attention_mask", "labels"}

        def collate_with_extras(features):
            pruned = []
            for f in features:
                pruned.append({k: v for k, v in f.items() if k in _MODEL_KEYS})
            batch = base_collator(pruned)
            return batch

        default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r = getattr(self.config.lora, 'r', 16)
        lora_alpha = getattr(self.config.lora, 'alpha', 32)
        lora_dropout = getattr(self.config.lora, 'dropout', 0.05)
        lora_target = list(getattr(self.config.lora, 'target_modules', default_target_modules))
        print(f"[TaskTrainer] LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"[TaskTrainer] LoRA target_modules: {lora_target}")

        peft_config = LoraConfig(
            base_model_name_or_path=self.config.model.id,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target,
        )

        # Metrics wiring
        preprocess_logits_for_metrics = make_preprocess_logits_for_metrics(self.factory.target_token_ids)
        compute_metrics = make_compute_metrics(self.factory.target_token_ids)

        # Only use validation for eval (test has no labels)
        eval_dataset = {"validation": self.validation_dataset}

        # --- Warm-start from a previous checkpoint if load_checkpoint is set ---
        load_checkpoint = getattr(self.config.model, 'load_checkpoint', None)
        if load_checkpoint and os.path.isdir(str(load_checkpoint)):
            print(f"[TaskTrainer] Warm-starting from checkpoint: {load_checkpoint}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.id,
                token=self.hugging_token,
                cache_dir=self.config.model.cache_dir,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            fixed_path, needs_adapter_cleanup = fix_adapter_key_prefix(str(load_checkpoint))
            model_to_train = PeftModel.from_pretrained(base_model, fixed_path, is_trainable=True)
            if needs_adapter_cleanup:
                shutil.rmtree(fixed_path, ignore_errors=True)
            model_to_train.print_trainable_parameters()
            peft_config = None  # Already has LoRA applied
        else:
            if load_checkpoint:
                print(f"[TaskTrainer] WARNING: load_checkpoint path not found: {load_checkpoint}")
            model_to_train = self.config.model.id  # TRL/Transformers will load the model
            # peft_config stays as the fresh LoraConfig

        # IMPORTANT for Accelerate/DDP:
        # Omit device_map="auto" and let Accelerate place devices.
        trainer = SALSATrainer(
            target_token_ids=self.factory.target_token_ids,
            balanced_sampler=self.balanced_sampler,
            model=model_to_train,
            processing_class=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_with_extras,
            peft_config=peft_config,
            args=SFTConfig(
                max_length=self.max_seq_length,  # Truncate sequences to this length
                model_init_kwargs=dict(
                    max_memory=self.config.model.max_memory,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    token=self.hugging_token,
                    cache_dir=self.config.model.cache_dir,
                ),
                max_grad_norm=1.0,
                deepspeed=getattr(self.config.training, 'deepspeed_config', None),
                remove_unused_columns=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                ddp_find_unused_parameters=False,
                ddp_timeout=7200,  # 2 hours; generous timeout for test inference at checkpoints

                per_device_eval_batch_size=getattr(self.config.training, 'per_device_eval_batch_size', 2),
                per_device_train_batch_size=getattr(self.config.training, 'per_device_train_batch_size', 2),
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=self.config.training.epochs,
                learning_rate=self.config.training.learning_rate,
                bf16=True,
                fp16=False,

                # Logging / W&B
                report_to="wandb",
                logging_steps=10,

                # Eval & checkpointing
                eval_strategy="steps",
                eval_steps=self.config.logging.save_steps.eval_step,
                save_strategy="steps",
                save_steps=self.config.logging.save_steps.eval_step,
                save_total_limit=self.config.training.save_total_limit,

                # Use fused optimizer when not using DeepSpeed (faster on CUDA)
                optim="adamw_torch" if getattr(self.config.training, 'deepspeed_config', None) else "adamw_torch_fused",
                weight_decay=self.config.training.weight_decay,
                lr_scheduler_type="linear",
                seed=self.config.training.seed,
                output_dir=self.config.output.checkpoint_path,

                dataloader_drop_last=True,

                # Resume behavior: skip already-seen data when resuming from checkpoint
                ignore_data_skip=False,
            ),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add test inference callback if test dataset is available
        if self.test_dataset is not None:
            inference_callback = TestInferenceCallback(
                test_dataset=self.test_dataset,
                test_ids=self.test_ids,
                target_token_ids=self.factory.target_token_ids,
                output_dir=self.config.output.checkpoint_path,
                tokenizer=tokenizer,
                batch_size=2,  # Match eval batch size
                test_labels=self.test_labels,
            )
            trainer.add_callback(inference_callback)
            has_labels_msg = " (with F1 logging)" if self.test_labels is not None else ""
            print(f"[TaskTrainer] Test inference callback added{has_labels_msg}. Will generate submissions at each checkpoint.")

        # Check for existing checkpoints if resume is enabled
        resume_from_checkpoint = None
        if getattr(self.config.training, 'resume', True):
            checkpoint_dir = self.config.output.checkpoint_path
            if os.path.isdir(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    resume_from_checkpoint = True  # HF Trainer auto-finds latest
                    # Find the latest checkpoint to show what step we're resuming from
                    checkpoint_steps = [int(c.split("-")[1]) for c in checkpoints if c.split("-")[1].isdigit()]
                    if checkpoint_steps:
                        latest_step = max(checkpoint_steps)
                        print(f"[TaskTrainer] Resuming from checkpoint in {checkpoint_dir}")
                        print(f"[TaskTrainer] Latest checkpoint: checkpoint-{latest_step}")
                        print(f"[TaskTrainer] Data skipping enabled (ignore_data_skip=False): "
                              f"will skip batches up to step {latest_step}")
                    else:
                        print(f"[TaskTrainer] Resuming from checkpoint in {checkpoint_dir}")
                else:
                    print(f"[TaskTrainer] No checkpoints found in {checkpoint_dir}, starting fresh.")
        else:
            print(f"[TaskTrainer] Resume disabled, starting fresh training.")

        # Train (HF Trainer will restore state and skip data automatically when resuming)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log final state after training completes
        print(f"[TaskTrainer] Training completed at epoch {trainer.state.epoch:.4f}, "
              f"global_step {trainer.state.global_step}")

        # Final validation evaluation
        final_val_metrics = trainer.evaluate(self.validation_dataset, metric_key_prefix="final_val")
        print("Final validation metrics:", final_val_metrics)

        # Summarize best model on W&B
        best_metric = trainer.state.best_metric
        best_ckpt = trainer.state.best_model_checkpoint
        print("Best ROC-AUC:", best_metric, "from", best_ckpt)

        # Run final test inference and save submission
        if self.test_dataset is not None:
            print("\n[TaskTrainer] Running final test inference...")
            final_callback = TestInferenceCallback(
                test_dataset=self.test_dataset,
                test_ids=self.test_ids,
                target_token_ids=self.factory.target_token_ids,
                output_dir=self.config.output.checkpoint_path,
                tokenizer=tokenizer,
                batch_size=2,
                test_labels=self.test_labels,
            )
            final_predictions = final_callback._run_inference(trainer.model)
            final_submission_path = os.path.join(self.config.output.checkpoint_path, "final_submission.csv")
            final_callback._save_submission(final_predictions, final_submission_path)
            print(f"[TaskTrainer] Saved final submission to {final_submission_path}")

            # Compute and log final F1 if labels available
            if self.test_labels is not None:
                try:
                    final_f1 = f1_score(self.test_labels, final_predictions, average="macro", zero_division=0)
                    print(f"[TaskTrainer] Final test F1 (macro): {final_f1:.4f}")
                    wandb.log({"test/final_f1": final_f1}, step=trainer.state.global_step)
                except Exception as e:
                    print(f"[TaskTrainer] Failed to compute final F1: {e}")

    def evaluate(self, dataset=None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.id,
            token=self.hugging_token,
            cache_dir=self.config.model.cache_dir,
        )

        base_collator = DataCollatorForSeq2Seq(tokenizer)

        _EVAL_MODEL_KEYS = {"input_ids", "attention_mask", "labels"}

        def collate_with_extras(features):
            pruned = [{k: v for k, v in f.items() if k in _EVAL_MODEL_KEYS} for f in features]
            return base_collator(pruned)

        preprocess_logits_for_metrics = make_preprocess_logits_for_metrics(self.factory.target_token_ids)
        compute_metrics = make_compute_metrics(self.factory.target_token_ids)

        eval_ds = dataset if dataset is not None else self.test_dataset or self.validation_dataset

        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        load_checkpoint = getattr(self.config.model, "load_checkpoint", None)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.id,
            trust_remote_code=True,
            torch_dtype="auto",
            token=self.hugging_token,
            cache_dir=self.config.model.cache_dir,
        )
        if load_checkpoint:
            fixed_path, needs_cleanup = fix_adapter_key_prefix(str(load_checkpoint))
            model = PeftModel.from_pretrained(model, fixed_path)
            if needs_cleanup:
                shutil.rmtree(fixed_path, ignore_errors=True)

        trainer = SALSATrainer(
            target_token_ids=self.factory.target_token_ids,
            model=model,
            processing_class=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=eval_ds,
            data_collator=collate_with_extras,
            args=SFTConfig(
                per_device_eval_batch_size=2,
                output_dir=self.config.output.checkpoint_path,
                remove_unused_columns=False,
                bf16=True,
                report_to="none",
            ),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        return trainer.evaluate(eval_ds)
