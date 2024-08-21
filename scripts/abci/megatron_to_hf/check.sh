#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:01:00:00
#$ -j y
#$ -o outputs/megatron-to-hf/
#$ -cwd

# Load modules
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

# swich virtual env
source .env/bin/activate

python scripts/abci/megatron_to_hf/check.py \
  --base-hf-model-path /bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-70B \
  --converted-hf-model-path /bb/llm/gaf51275/2024/checkpoints/megatron-to-hf/Llama-3.1-70b/checkpoint-convert-test/iter_0000001
