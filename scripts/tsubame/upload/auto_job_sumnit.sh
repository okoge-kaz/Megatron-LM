#!/bin/bash

start=1000
end=25000
increment=500

for ((i = start; i <= end; i += increment)); do
  qsub -g hp190122 scripts/tsubame/upload/upload_Llama-3-70b-megatron.sh $i $i
done
