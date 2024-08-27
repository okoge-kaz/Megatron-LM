#!/bin/bash

set -e

start=0
end=99

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python scripts/abci/upload/upload_dataset.py \
        --ckpt-path "$upload_dir" \
        --repo-name "$repo_name" \
        --start-range "$start" \
        --end-range "$end"; then
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

upload_dir=/groups/gag51395/vlm_datasets/OBELICS
repo_name="RioYokotaLab/OBELICS-$(printf "%07d" $start)-$(printf "%07d" $end)"

if ! upload_checkpoint "$upload_dir" "$repo_name"; then
  echo "Skipping to next checkpoint after repeated failures for $repo_name"
fi
