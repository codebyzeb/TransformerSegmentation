
MODELS=(gpt2_85M)

for model in ${MODELS[@]};
do
    echo "Launching job with model: $model with text BPE tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model experiment.name=$model-bpe-txt
    echo "Launching job with model: $model with text BPE tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_bpe_spaceless experiment.name=$model-bpe-txt-spaceless
    echo "Launching job with model: $model with character tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_char experiment.name=$model-char-txt
    echo "Launching job with model: $model with character tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_char_spaceless experiment.name=$model-char-txt-spaceless

    echo "Launching job with model: $model with phoneme BPE tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_bpe experiment.name=$model-bpe-phoneme
    echo "Launching job with model: $model with phoneme BPE tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_bpe_spaceless experiment.name=$model-bpe-phoneme-spaceless
    echo "Launching job with model: $model with phoneme tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme experiment.name=$model-phoneme
    echo "Launching job with model: $model with phoneme tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_spaceless experiment.name=$model-phoneme-spaceless
done