#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/check/$JOB_ID.log
#$ -e outputs/convert/check/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# switch virtual env
source .env/bin/activate

python scripts/tsubame/chpt-tests/check.py \
  --base-hf-model-path /gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B \
  --converted-hf-model-path /gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3.1-8b-v0.8/tp2-pp2-ct1/iter_0000001
