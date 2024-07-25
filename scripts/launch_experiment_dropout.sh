DATA_SIZES=(10000 25000 50000 100000 250000 500000 1000000)
DROPOUTS=(0.1 0.3 0.5 0.7)
MODEL="gpt2_600k"

for data_size in ${DATA_SIZES[@]};
do
    for dropout in ${DROPOUTS[@]};
    do
        echo "Launching job for data size: $data_size and dropout: $dropout"
        sbatch launch_slurm.wilkes3 experiment.group=dropout experiment.evaluate_babyslm=True tokenizer.name=transformersegmentation/CHILDES-English-phoneme-tokenizer dataset.num_examples=$data_size model=$MODEL +model.attn_pdrop=$dropout +model.resid_pdrop=$dropout +model.embd_pdrop=$dropout $@
    done
done
