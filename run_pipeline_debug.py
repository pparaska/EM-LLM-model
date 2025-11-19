import os
import sys
import subprocess

# Set these as needed for your debugging session
MODEL = "mistral"
BENCHMARK = "long-bench"
DATASET = os.environ.get("EMLLM_DATASET", "2wikimqa")  # Change as needed
ALLOW_DISK_OFFLOAD = "True"
WORLD_SIZE = "1"
NUM_GPUS_PER_JOB = "1"
RANK_OFFSET = "0"

# Compute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_PATH = os.path.join(BASE_DIR, "benchmark", "results", MODEL, BENCHMARK)
CONFIG_FILE = f"{MODEL}.yaml"
CONFIG_PATH = os.path.join(BASE_DIR, "config", CONFIG_FILE)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force HuggingFace to use cached models only (no download/update checks)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Build pred.py command
pred_cmd = [
    sys.executable, os.path.join(BASE_DIR, "benchmark", "pred.py"),
    "--config_path", CONFIG_PATH,
    "--output_dir_path", OUTPUT_DIR_PATH,
    "--datasets", DATASET,
    "--world_size", WORLD_SIZE,
    "--rank", "0",
    "--allow_disk_offload", ALLOW_DISK_OFFLOAD
]

print("Running:", " ".join(pred_cmd))
subprocess.run(pred_cmd, check=True)

# Clean up offload data directory if it exists
offload_dir = os.path.join(OUTPUT_DIR_PATH, "offload_data")
if os.path.isdir(offload_dir):
    import shutil
    shutil.rmtree(offload_dir)
    print(f"Deleted offload data directory: {offload_dir}")

# Build eval.py command

eval_cmd = [
    sys.executable, os.path.join(BASE_DIR, "benchmark", "eval.py"),
    "--dir_path", OUTPUT_DIR_PATH
]

print("Running:", " ".join(eval_cmd))
subprocess.run(eval_cmd, check=True)

print("Pipeline completed.")
