#!/usr/bin/env python
"""
Preprocess the HatExplain dataset into local parquet files.

The original HF dataset (Hate-speech-CNERG/hatexplain) uses a deprecated
dataset script and the previously-used processed version (bsvaz/hatexplain-processed)
is no longer available. This script downloads the raw JSON and produces
train/validation/test parquets with columns: text, label.

Labels: 0=hatespeech, 1=normal, 2=offensive (majority vote from 3 annotators)

Usage:
    python scripts/preprocess_hatexplain.py
"""

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    output_dir = Path("datasets/hatexplain")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download raw JSON from the HF repo
    print("Downloading hatexplain dataset...")
    json_path = hf_hub_download(
        repo_id="Hate-speech-CNERG/hatexplain",
        filename="dataset.json",
        repo_type="dataset",
    )

    with open(json_path) as f:
        data = json.load(f)

    # Process each entry
    splits = {"train": [], "val": [], "test": []}
    for post_id, entry in data.items():
        # Join tokens into text
        text = " ".join(entry["post_tokens"])

        # Majority vote label from annotators
        labels = [a["label"] for a in entry["annotators"]]
        majority_label = Counter(labels).most_common(1)[0][0]

        # Map to split
        split = entry.get("split", "train")
        if split == "val":
            split_key = "val"
        elif split == "test":
            split_key = "test"
        else:
            split_key = "train"

        splits[split_key].append({"text": text, "label": majority_label})

    # Save parquets
    split_map = {"train": "train", "val": "validation", "test": "test"}
    for key, rows in splits.items():
        df = pd.DataFrame(rows)
        out_name = split_map[key]
        out_path = output_dir / f"{out_name}.parquet"
        df.to_parquet(out_path, index=False)
        label_dist = df["label"].value_counts().sort_index().to_dict()
        print(f"  {out_name}: {len(df)} rows -> {out_path} (labels: {label_dist})")

    print("Done!")


if __name__ == "__main__":
    main()
