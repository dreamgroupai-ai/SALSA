import os

import pandas as pd
from datasets import Dataset, Value, load_dataset
from transformers import AutoTokenizer


class SalsaDatasetFactory:
    def __init__(self, config, max_seq_length: int = 8192, num_proc: int = None):
        self.config = config
        self.max_seq_length = max_seq_length
        # Number of workers for dataset mapping (default: CPU count)
        self.num_proc = num_proc if num_proc is not None else min(16, os.cpu_count() or 1)
        self.hugging_token = os.environ.get("HuggingfaceToken")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.id,
            token=self.hugging_token,
            cache_dir=self.config.model.cache_dir,
        )
        # Set truncation to left side to preserve labels at the end of sequences
        self.tokenizer.truncation_side = "left"
        # list of class label strings as they appear in the assistant's single-token answer
        self.classes = self.config.classifier.options["label"]
        # map class strings to single target token ids
        self.target_token_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in self.classes]
        self.eot_ids = None  # set in compile_dataset

    def _rename_fields(self, hf_dataset: Dataset) -> Dataset:
        """Rename dataset columns so prompt placeholders match.

        Uses the optional ``fields`` mapping from the dataset config.
        Each entry maps ``placeholder_name: dataset_column_name``.
        Identity mappings (where placeholder == column) are skipped.
        """
        fields = getattr(self.config.dataset, "fields", None)
        if not fields:
            return hf_dataset
        for placeholder, column in fields.items():
            if placeholder != column and column in hf_dataset.column_names:
                hf_dataset = hf_dataset.rename_column(column, placeholder)
        return hf_dataset

    def _load_dataset(self, split) -> Dataset:
        """Load dataset from local parquet or HuggingFace."""
        config = self.config
        local_path = getattr(config.dataset, "local_path", None)

        if local_path and os.path.exists(local_path):
            # Load from local parquet file
            parquet_file = os.path.join(local_path, f"{split}.parquet")
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                return Dataset.from_pandas(df, preserve_index=False)
            else:
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
        else:
            # Fall back to HuggingFace
            if isinstance(config.dataset.name, str):
                return load_dataset(
                    config.dataset.name,
                    split=split,
                    trust_remote_code=False,
                )
            else:
                return load_dataset(
                    *config.dataset.name,
                    split=split,
                    trust_remote_code=False,
                )

    def compile_dataset(self, split) -> Dataset:
        """Compile dataset for training/validation (with labels)."""
        config = self.config
        EOT_MARKER = "<|im_end|>"  # or take from config if needed

        hf_dataset = self._load_dataset(split)
        hf_dataset = self._rename_fields(hf_dataset)

        # Check if this is inference mode (no labels)
        is_inference = "label" not in hf_dataset.column_names

        if is_inference:
            # Test set: no label remapping, inference mode
            return self._compile_inference_dataset(hf_dataset)

        # Preserve original label for balanced sampling before remapping
        hf_dataset = hf_dataset.map(
            lambda x: {"original_label": x["label"]}, num_proc=self.num_proc
        )

        # Remap raw labels -> target label tokens (strings)
        # Cast numeric label columns to float64 to avoid ClassLabel validation errors
        # (e.g. medmcqa has ClassLabel typed column that rejects string values).
        # String labels (e.g. mednli's "entailment") are left as-is.
        labels_are_numeric = all(isinstance(l, (int, float)) for l in config.dataset.labels)
        if labels_are_numeric:
            hf_dataset = hf_dataset.cast_column("label", Value("float64"))
        mapping = dict(zip(config.dataset.labels, self.config.classifier.options["label"]))
        # round() handles continuous float labels like stsb (0.0-5.0 -> nearest int);
        # no-op on int labels and skipped entirely for string labels
        if labels_are_numeric:
            hf_dataset = hf_dataset.map(lambda x: {"label": mapping[round(x["label"])]}, num_proc=self.num_proc)
        else:
            hf_dataset = hf_dataset.map(lambda x: {"label": mapping[x["label"]]}, num_proc=self.num_proc)

        # Get balance columns from config (default: label and language)
        balance_columns = getattr(config.dataset, "balance_columns", ["original_label", "language"])

        dataset = self._compile_dataframe(hf_dataset, balance_columns)
        dataset = self._formatting(dataset, balance_columns)
        # store EOT id(s)
        self.eot_ids = [self.tokenizer.convert_tokens_to_ids(EOT_MARKER)]

        return dataset.map(
            lambda x: self.custom_train_on_last_response_token(x, balance_columns=balance_columns),
            num_proc=self.num_proc
        )

    def compile_dataset_with_ids(self, split):
        """
        Compile dataset and return (dataset, ids) tuple.
        Used for test set where we need to preserve IDs for submission.
        """
        config = self.config
        hf_dataset = self._load_dataset(split)
        hf_dataset = self._rename_fields(hf_dataset)

        # Extract IDs if present (for test set submission)
        ids = None
        if "ID" in hf_dataset.column_names:
            ids = hf_dataset["ID"]

        # Check if this is inference mode (no labels)
        is_inference = "label" not in hf_dataset.column_names

        if is_inference:
            dataset = self._compile_inference_dataset(hf_dataset)
            return dataset, ids
        else:
            # For train/val, use regular compile (note: dataset loaded twice,
            # but compile_dataset also applies _rename_fields which is idempotent)
            dataset = self.compile_dataset(split)
            return dataset, ids

    def get_test_labels(self, split):
        """
        Get ground truth labels for the test split if available.
        Returns list of labels or None if labels don't exist.
        """
        try:
            hf_dataset = self._load_dataset(split)
            hf_dataset = self._rename_fields(hf_dataset)
            if "label" in hf_dataset.column_names:
                return hf_dataset["label"]
            return None
        except Exception as e:
            print(f"[SalsaDatasetFactory] Could not load test labels: {e}")
            return None

    def _compile_inference_dataset(self, hf_dataset) -> Dataset:
        """Compile dataset for inference (no labels, user prompt only with generation prompt)."""
        # Build conversations with user prompt only
        dataset = self._compile_dataframe_inference(hf_dataset)
        dataset = self._formatting_inference(dataset)
        return dataset.map(self._tokenize_for_inference, num_proc=self.num_proc)

    def _compile_dataframe_inference(self, df):
        """Build conversations for inference (user prompt only, no assistant answer)."""
        dataset = []
        for record in df:
            conversation = [
                {"role": "user", "content": self.config.classifier.main_prompt.format(**record)},
            ]
            # Preserve ID if present
            sample_id = record.get("ID", None)
            dataset.append((conversation, sample_id))
        return dataset

    def _formatting_inference(self, dataset):
        """Format inference dataset with generation prompt + thinking wrapper + answer prefix."""
        texts, ids = [], []
        # Get the answer prefix from config (e.g., "<ANSWER>")
        answer_prefix = getattr(self.config.classifier, 'prefix', '')
        for i in range(len(dataset)):
            # Apply chat template with generation prompt and enable_thinking=False
            # This adds <think>\n\n</think>\n\n to match training format
            text = self.tokenizer.apply_chat_template(
                dataset[i][0], tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            # Append answer prefix so model predicts label at the right position
            texts.append(text + answer_prefix)
            ids.append(dataset[i][1])
        return Dataset.from_dict({"text": texts, "sample_id": ids})

    def _tokenize_for_inference(self, example):
        """Tokenize for inference (no labels needed)."""
        tokenized = self.tokenizer(
            example["text"],
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = tokenized["input_ids"][0]

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": [1] * len(input_ids),
            "sample_id": example.get("sample_id"),
        }

    def _compile_dataframe(self, df, balance_columns=None):
        # Build conversations and attach optional per-sample scores/weights if present
        # Also preserve balance columns for stratified sampling
        balance_columns = balance_columns or []
        dataset = []
        for record in df:
            conversation = [
                {"role": "user", "content": self.config.classifier.main_prompt.format(**record)},
                {"role": "assistant", "content": self.config.classifier.answer_prompt.format(**record)},
            ]
            # Extract balance column values
            balance_values = {col: record.get(col, None) for col in balance_columns}
            dataset.append((
                conversation,
                record.get("probs", -1),
                1,  # default sample weight
                balance_values,
            ))
        return dataset

    def _formatting(self, dataset, balance_columns=None):
        balance_columns = balance_columns or []
        texts, probs, weight = [], [], []
        # Initialize dict for balance column values
        balance_data = {col: [] for col in balance_columns}

        for i in range(len(dataset)):
            texts.append(
                self.tokenizer.apply_chat_template(
                    dataset[i][0], tokenize=False, add_generation_prompt=False,
                    enable_thinking=False
                )
            )
            probs.append(dataset[i][1])
            weight.append(dataset[i][2])
            # Extract balance column values from tuple
            balance_values = dataset[i][3] if len(dataset[i]) > 3 else {}
            for col in balance_columns:
                balance_data[col].append(balance_values.get(col, None))

        result_dict = {"text": texts, "probs": probs, "weight": weight}
        result_dict.update(balance_data)
        return Dataset.from_dict(result_dict)

    def _find_last_subsequence(self, haystack_ids, needle_ids):
        n = len(needle_ids)
        if n == 0:
            return -1
        for i in range(len(haystack_ids) - n, -1, -1):
            if haystack_ids[i : i + n] == needle_ids:
                return i
        return -1

    def _find_last_eot_index(self, input_ids_tensor, eot_ids):
        ids = input_ids_tensor.tolist()
        last = -1
        for eot in eot_ids:
            if isinstance(eot, (list, tuple)):
                idx = self._find_last_subsequence(ids, list(eot))
            else:
                try:
                    idx = len(ids) - 1 - ids[::-1].index(eot)
                except ValueError:
                    idx = -1
            if idx > last:
                last = idx
        return last

    def custom_train_on_last_response_token(self, example, mode="learn", balance_columns=None):
        # Tokenize full chat text with truncation (left-side to preserve label at end)
        tokenized = self.tokenizer(
            example["text"],
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = tokenized["input_ids"][0]

        # -100 everywhere except the *last* assistant label token
        labels = [-100] * len(input_ids)

        # For single-label classification, the assistant answer contains a single class token.
        # We set the last EOT (or marker) position as the supervised token.
        # If you want the class token position exactly, replace with a search for that token.
        label_pos = self._find_last_eot_index(input_ids, self.target_token_ids)
        if label_pos >= 0:
            labels[label_pos] = int(input_ids[label_pos])

        result = {
            "input_ids": input_ids.tolist(),
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "mode": int(mode == "learn"),
            "probs": example.get("probs", -1),
            "weight": example.get("weight", 0.1),
        }

        # Preserve balance columns for stratified sampling
        balance_columns = balance_columns or []
        for col in balance_columns:
            if col in example:
                result[col] = example[col]

        return result
