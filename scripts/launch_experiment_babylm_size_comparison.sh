#MODELS=("gpt2_600k" "gpt2_1M" "gpt2_19M" "gpt2_85M")
MODELS=("gpt2_600k" "gpt2_1M" "gpt2_19M")
DATA_SIZES=(100000 1000000 10000000 95334997)

for data_size in ${DATA_SIZES[@]};
do
    for model in ${MODELS[@]};
    do
        echo "Launching jobs for data size: $data_size and model: $model"
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_01 model=$model data_preprocessing.subsample=$data_size experiment.name=$model-$data_size-01 $@
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_03 model=$model data_preprocessing.subsample=$data_size experiment.name=$model-$data_size-03 $@
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_05 model=$model data_preprocessing.subsample=$data_size experiment.name=$model-$data_size-05 $@
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_07 model=$model data_preprocessing.subsample=$data_size experiment.name=$model-$data_size-07 $@
    done
done
