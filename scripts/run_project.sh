#!/bin/bash
echo "Running language model training script"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate char_transformer
CMD="python run_model.py $@"
eval $CMD
echo "Script complete"
