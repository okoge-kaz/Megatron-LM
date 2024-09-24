#!/bin/sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=24:00:00
#$ -o outputs/upload/Llama-3.1-70B-megatron/$JOB_ID.log
#$ -e outputs/upload/Llama-3.1-70B-megatron/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

if [ $# -lt 2 ]; then
  echo "Usage: $0 <start_iteration> <end_iteration>"
  exit 1
fi

start=$1
end=$2
increment=500

upload_base_dir=/gs/bs/tga-NII-LLM/Llama-3.1-70B/tp4-pp8-ct1-LR1.0E-5-MINLR1.0E-6-WD0.1

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python scripts/abci/upload/upload.py \
        --ckpt-path "$upload_dir" \
        --repo-name "$repo_name"; then
        echo "Successfully uploaded $repo_name"
        return 0
    else
        echo "Upload failed for $repo_name. Retrying..."
        ((retry_count++))
        sleep 5
    fi
  done

  echo "Failed to upload $repo_name after $max_retries attempts"
  return 1
}

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  repo_name="RioYokotaLab/Llama-3.1-70b-LR1.0e-5-MINLR1.0E-6-WD0.1-iter$(printf "%07d" $i)"

  if ! upload_checkpoint "$upload_dir" "$repo_name"; then
    echo "Skipping to next checkpoint after repeated failures for $repo_name"
  fi
done
