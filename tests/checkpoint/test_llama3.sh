#!/bin/bash
#SBATCH --job-name=check
#SBATCH --time=24:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/check/%x-%j.out
#SBATCH --error=outputs/check/%x-%j.out

set -e

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

pytest -sv tests/checkpoint/tests_llama3.py
