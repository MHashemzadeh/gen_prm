import os
import jsonlines
from typing import List

def aggregate_accuracy(directory: str) -> float:
    total_count = 0
    correct_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            
            # Read each .jsonl file
            with jsonlines.open(file_path) as reader:
                for line in reader:
                    total_count += 1
                    if line == 1:
                        correct_count += 1

    # Compute and return the accuracy
    if total_count == 0:
        return 0.0
    else:
        return correct_count / total_count

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    accuracy = aggregate_accuracy(directory_path)
    print(f"Aggregate accuracy across {len([f for f in os.listdir(directory_path) if f.endswith('.jsonl')])} workers: {accuracy:.4f}")
