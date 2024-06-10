#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=7:00:00:00
#$ -j y
#$ -o outputs/fineweb-edu/
#$ -cwd

set -e

# swich virtual env
source .env/bin/activate

DATASET_DOWNLOAD_DIR=/groups/gag51395/datasets/raw
cd $DATASET_DOWNLOAD_DIR

# Git LFS
git lfs install
git clone git@hf.co:datasets/HuggingFaceFW/fineweb-edu
