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
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

# below environment variables are for greedy decoding test
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=1

# can not pass the test (decoding is not deterministic)
pytest -sv tests/checkpoint/tests_decoding_llama2.py
