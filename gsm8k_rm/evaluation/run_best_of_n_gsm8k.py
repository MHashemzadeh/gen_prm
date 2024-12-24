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
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.8,
    n_samples: int = 40,
    batch_size: int = 128,
    prm_bsz: int = 40,
    data_split: str = "test",
    aggregate_method: str = "mean",
    seed: int = 42,
    use_baseline_prm: bool = False,
):
    # Configuration grid
    # prm_configs = {
    #     'prm_types': ['segregated', 'segregated', 'segregated'],
    #     'prm_paths': [
    #         'outputs/trained_models/prm/gen/segregated/llama3.1-8b-instruct/winter-dew-22/best_model/' # CoT with scaling
    #         #'outputs/trained_models/prm/gen/segregated-direct/best_model/',
    #         #'outputs/trained_models/prm/gen/segregated/llama3.1-8b-instruct/winter-dew-22/best_model/', # CoT
    #     ],
    #     'prm_ns': [1, 1, 10],
    #     'prm_temperatures': [0.0, 0.0, 0.4]
    # }

        # Configuration grid
    prm_configs = {
        'prm_types': ['segregated', 'segregated'],
        'prm_paths': [
            #'outputs/trained_models/magic-serenity-43/best_model/',
            'outputs/trained_models/magic-serenity-43/iteration_2/', # CoT - balanced
            #'outputs/trained_models/prm/gsm8k/prm-and-sampler/sweet-lake-39/checkpoint-14000/', # CoT - balanced'
            #'outputs/trained_models/prm/gsm8k/prm-and-sampler/sweet-lake-39/best_model/'
            #'outputs/trained_models/prm/all_math/self-taught/segregated/llama3.1-8b-instruct/fragrant-brook-32/best_model/', # CoT
        ],
        'prm_ns': [1, 1],
        'prm_temperatures': [0.0, 0.0]
    }

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create results directory
    results_dir = f"outputs/results/gsm8k/"
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments for all configurations
    results = []
    for prm_type, prm_path, prm_n, prm_temp in zip(
        prm_configs['prm_types'],
        prm_configs['prm_paths'], 
        prm_configs['prm_ns'],
        prm_configs['prm_temperatures']
    ):
        print(f"\nRunning experiment with:")
        print(f"PRM Type: {prm_type}")
        print(f"PRM Path: {prm_path}")
        print(f"PRM N: {prm_n}")
        print(f"PRM Temperature: {prm_temp}")

        # Set environment variables
        os.environ["SEED"] = str(seed)
        os.environ["PYTHONPATH"] = "."
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["RAY_DEDUP_LOGS"] = "0"
        
        # num_instances is the number of gpus available
        
        num_instances = torch.cuda.device_count()

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
            "--prm_path", prm_path,
            "--prm_bsz", str(prm_bsz),
            "--prm_type", prm_type,
            "--prm_n", str(prm_n),
            "--prm_temperature", str(prm_temp),
            "--num_instances", str(num_instances),
        ]
        
        if use_baseline_prm:
            cmd.append("--use_baseline_prm")

        print(f"Running command: {' '.join(cmd)}")

        # Import subprocess at top of file
        import subprocess
        
        # Run process and capture output
        result = subprocess.run(" ".join(cmd), shell=True, text=True)
        output = result.stdout

        # Parse accuracy from output
        for line in output.split('\n')[::-1]: # reverse order to get final accuracy
            if line.startswith('Final Accuracy:') or line.startswith('Accuracy:'):
                accuracy = float(line.split(':')[1].strip())
                break
        
        print(f"Accuracy: {accuracy}")

        result = {
            'prm_type': prm_type,
            'prm_path': prm_path,
            'prm_n': prm_n,
            'prm_temperature': prm_temp,
            'accuracy': accuracy,
            'temperature': temperature,
            'n_samples': n_samples,
            'aggregate_method': aggregate_method
        }
        results.append(result)

        # ### if exists, append to file
        # if os.path.exists(f"{results_dir}/best_of_n.json"):
        #     with open(f"{results_dir}/best_of_n.json", 'a') as f:
        #         json.dump(result, f, indent=2)
        # else:
        #     # Save intermediate results
        #     with open(f"{results_dir}/best_of_n.json", 'w') as f:
        #         json.dump(results, f, indent=2)

    # Print final summary
    print("\nFinal Results Summary:")
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"\nAccuracy: {result['accuracy']:.4f}")
        print(f"PRM Type: {result['prm_type']}")
        print(f"PRM N: {result['prm_n']}")
        print(f"PRM Temperature: {result['prm_temperature']}")

if __name__ == "__main__":
    fire.Fire(main)
