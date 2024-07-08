LAYERS=(2 3 4 6 6 6 8 12)
HEADS=(4 4 4 4 8 8 8 12)
HIDDEN=(128 128 128 128 256 512 512 768)
INNER=(512 512 512 512 1024 2048 2048 3072)

for i in {0..7}
do
    sbatch launch_slurm.wilkes3 model.n_layer=${LAYERS[i]} model.n_head=${HEADS[i]} model.n_embd=${HIDDEN[i]} model.n_inner=${INNER[i]} $@
done
