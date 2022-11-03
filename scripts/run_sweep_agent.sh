#!/bin/bash
if [ -z $1 ]; then
    echo "No sweep ID provided"
    exit 1
fi
echo "Running a single run as part of sweep with ID: \"$1\""
source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer_seg

CMD="wandb agent --count 1 $1"
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
