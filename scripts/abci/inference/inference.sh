#!/bin/bash
#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=03:30:00
#$ -j y
#$ -o outputs/inference/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

source .env/bin/activate

# inference
python scripts/abci/inference/inference.py \
  --hf-model-path "/bb/llm/gaf51275/2024/checkpoints/megatron-to-hf/Llama-3.1-8b/datacom-lm/tp2-pp4-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1/iter_0002500"
