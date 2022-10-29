#!/bin/bash
if [ -z $1 ]; then
    echo "No sweep ID provided"
    exit 1
fi
echo "Running a single run as part of sweep with ID: \"$1\""
source ~/miniconda3/etc/profile.d/conda.sh
conda activate char_transformer
CMD="wandb agent --count 1 $1"
eval $CMD
echo "Script complete"
