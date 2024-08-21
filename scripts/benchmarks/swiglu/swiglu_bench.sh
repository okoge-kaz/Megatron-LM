#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/megatron_hf/$JOB_ID.log
#$ -e outputs/convert/megatron_hf/$JOB_ID.log
#$ -p -3

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# switch virtual env
source .env/bin/activate

# run
python scripts/benchmarks/swiglu.py
