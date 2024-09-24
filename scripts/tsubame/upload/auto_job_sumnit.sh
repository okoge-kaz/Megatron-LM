#!/bin/bash

start=1500
end=15000
increment=500

for ((i = start; i <= end; i += increment)); do
  qsub -g hp190122 scripts/tsubame/upload/upload_llama3.1_70b.sh $i $i
done
