MODELS=("gpt2_400k" "gpt2_600k" "gpt2_800k" "gpt2_1M" "gpt2_5M" "gpt2_19M" "gpt2_25M" "gpt2_85M")

for i in {0..7}
do
    sbatch launch_slurm.wilkes3 model=${MODELS[i]} $@
done
