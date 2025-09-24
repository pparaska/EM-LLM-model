#!/usr/bin/env bash

SHORT=m:,b:,w:,n:,r:,o:,d:,h:
LONG=model:,benchmark:,world_size:,num_gpus_per_job:,rank_offset:,allow_disk_offload:,dataset:,help:

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

# Default values
model=mistral 
benchmark=long-bench 
num_gpus_per_job=1 
rank_offset=0 
allow_disk_offload=False
dataset=""

nvidia-smi
world_size=$(nvidia-smi --list-gpus | wc -l)
echo "Visible GPUs: $world_size"

while true; do
    case "$1" in
        -h|--help) echo "Usage: $0 [--model <str>] [--benchmark <str>] [--dataset <str>] [--world_size <int>] [--num_gpus_per_job <int>] [--rank_offset <int>] [--allow_disk_offload <bool>] [--help]"; exit ;;
        -m|--model) model="$2"; shift 2 ;;
        -b|--benchmark) benchmark="$2"; shift 2 ;;
        -w|--world_size) world_size="$2"; shift 2 ;;
        -n|--num_gpus_per_job) num_gpus_per_job="$2"; shift 2 ;;
        -r|--rank_offset) rank_offset="$2"; shift 2 ;;
        -o|--allow_disk_offload) allow_disk_offload="$2"; shift 2 ;;
        -d|--dataset) dataset="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Programming error"; exit 3 ;;
    esac
done

# Check dataset
if [ -z "$dataset" ]; then
    echo "Please provide a dataset with --dataset <name> (e.g., --dataset 2wikimqa)"
    exit 1
fi

# ============================ SETUP ============================ #
base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$base_dir"

echo "World size: $world_size"

output_dir_path="${base_dir}/benchmark/results/${model}/${benchmark}"
config_file="${model}.yaml"

echo "--------------------------------------------------"
echo "Starting evaluation on dataset: $dataset"
echo "Using model: $model"
echo "Benchmark: $benchmark"
echo "Disk offload: $allow_disk_offload"
echo "Output directory: $output_dir_path"
echo "--------------------------------------------------"

# Enable memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================ RUN ============================ #
CUDA_VISIBLE_DEVICES=0 python "${base_dir}/benchmark/pred.py" \
    --config_path "${base_dir}/config/${config_file}" \
    --output_dir_path "${output_dir_path}" \
    --datasets "$dataset" \
    --world_size "1" \
    --rank "0" \
    --allow_disk_offload "${allow_disk_offload}"

echo "Completed: $dataset"

# Clean up GPU/disk cache to avoid OOM
DIRECTORY="${output_dir_path}/offload_data"
if [ -d "$DIRECTORY" ]; then
    rm -rf "$DIRECTORY"
    echo "Deleted offload data directory: $DIRECTORY"
fi

# Optional: Evaluate
python "${base_dir}/benchmark/eval.py" --dir_path "${output_dir_path}"
