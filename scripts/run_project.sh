#!/bin/bash
echo "Running language model training script"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer_seg
CMD="python run_model.py $@"
eval $CMD

# Deal with requeuing
if [[ $? -eq 124 ]]; then
  echo "Encountered error code 124. Trying to requeue by calling tmp/requeue.sh"
  eval "tmp/requeue.sh"
  eval "rm tmp/requeue.sh"
else
  echo "Script complete with no error code"
fi

echo "Script complete"
