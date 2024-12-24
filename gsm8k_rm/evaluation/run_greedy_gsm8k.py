#!/usr/bin/env python
import os
import sys
import json
import warnings
import fire
import random
import numpy as np
import torch

def main(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature: float = 0.0,
    n_samples: int = 1,
    batch_size: int = 128,
    data_split: str = "test",
    aggregate_method: str = "mean",
    seed: int = 42,
):
  
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create results directory
    results_dir = f"outputs/results/gsm8k/"
    os.makedirs(results_dir, exist_ok=True)

    # Set environment variables
    os.environ["SEED"] = str(seed)
    os.environ["PYTHONPATH"] = "."

    # Construct command
    cmd = [
        "python", "best_of_n_with_prm.py",
        "--output_trace_in_each_iter",
        "--temperature", str(temperature),
        "--n_samples", str(n_samples),
        "--base_lm", "vllm",
        "--batch_size", str(batch_size),
        "--hf_path", model,
        "--data_split", data_split,
        "--aggregate_method", aggregate_method,
        "--disable_log",
    ]

    # Run process
    process = os.popen(" ".join(cmd))
    output = process.read()
    process.close()

    # Parse accuracy from output
    for line in output.split('\n'):
        if line.startswith('Accuracy:'):
            accuracy = float(line.split(':')[1].strip())

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    fire.Fire(main)
