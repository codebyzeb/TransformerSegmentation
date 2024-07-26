#DATA_SIZES=(10000 25000 50000 100000 250000 500000 1000000)
DATA_SIZES=(100000 1000000 10000000 100000000)
DROPOUTS=(0.1 0.3 0.5 0.7)
MODELS=(gpt2_600k)

for model in ${MODELS[@]};
do
    for data_size in ${DATA_SIZES[@]};
    do
        echo "Launching job with model: $model and data size: $data_size for dropout: 0.1"
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_01 data_preprocessing.subsample=$data_size model=$MODEL $@
        echo "Launching job with model: $model and data size: $data_size for dropout: 0.3"
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_03 data_preprocessing.subsample=$data_size model=$MODEL $@
        echo "Launching job with model: $model and data size: $data_size for dropout: 0.5"
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_05 data_preprocessing.subsample=$data_size model=$MODEL $@
        echo "Launching job with model: $model and data size: $data_size for dropout: 0.7"
        sbatch launch_slurm.wilkes3 experiment=dropout_babylm_07 data_preprocessing.subsample=$data_size model=$MODEL $@
    done
done