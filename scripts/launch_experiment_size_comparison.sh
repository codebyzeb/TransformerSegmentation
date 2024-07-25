MODELS=("gpt2_400k" "gpt2_600k" "gpt2_800k" "gpt2_1M" "gpt2_5M" "gpt2_19M" "gpt2_25M" "gpt2_85M")
DATA_SIZES=(10000 25000 50000 100000 250000 500000 1000000)

for data_size in ${DATA_SIZES[@]};
do
    for model in ${MODELS[@]};
    do
        echo "Launching jobs for data size: $data_size and model: $model"
        sbatch launch_slurm.wilkes3 experiment.group=size-comparison experiment.evaluate_babyslm=True tokenizer.name=transformersegmentation/CHILDES-English-phoneme-tokenizer model=$model dataset.num_examples=$data_size $@
    done
done
