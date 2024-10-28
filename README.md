# Phoneme Transformers

This repository contains training scripts, configs and analysis notebooks related to training and evaluating transformers trained on phonemes. This project is still in development.

## Installation

If you plan to use Weights and Biases for logging (i.e. not running in offline mode) you will also need to specify your wandb project to `.env` with the name `WANDB_ENTITY`. In order to interact with the hub, you need to generate read and write [access tokens](https://huggingface.co/docs/hub/security-tokens) from your hugging face account. Once generated, store these values as environment variables in `.env` with the names `HF_READ_TOKEN`, and `HF_WRITE_TOKEN`:

```
export WANDB_ENTITY=""
export HF_READ_TOKEN=""
export HF_WRITE_TOKEN=""
```

Then, to set up the project, run `./setup.sh`. This sets up the requirements and git hooks for automatic code formatting. Additionally, this script makes sure you are logged into WandB and Huggingface.

## Training

The main entry point for training is `train.py`, which uses a modified Huggingface Trainer to train a model. We use Hydra to specify configurations, including model parameters, with an architecture and dataset specified by the chosen configuration. E.g:

```
python train.py experiment=base_experiment experiment.name=my-experiment experiment.group=debug
```

### Evaluation

The project supports the [BabyLM evaluation pipeline](https://github.com/babylm/evaluation-pipeline-2024) by specifying the tasks to run during training. This requires downloading and placing the evaluation data in the main directory under `./evaluation_data/babylm_eval`. The project also supports the [BabySLM](https://github.com/babylm/evaluation-pipeline-2024) evaluation pipeline and custom word segmentation experiments. 

### Hyperparameter Tuning

Hyperparameter tuning can be conducted using a Weights & Biases sweep. To create a sweep, run the following:

```
wandb sweep --project PROJECT_NAME scripts/sweep.yaml
```

This will create a new sweep in Weights & Biases with an ID and a URL. You can either run the agent by either:
1. Running an agent directly using wandb: `wandb agent USER_ID/PROJECT_NAME/SWEEP_ID`
2. Run the script provided: `sh scripts/run_sweep_agent USER_ID/PROJECT_NAME/SWEEP_ID`
3. Queue a SLURM job to run the agent on a different node: `sbatch scripts/slurm_submit_sweep_agent.wilkes3 USER_ID/PROJECT_NAME/SWEEP_ID`.

The first option starts a `wandb` agent on your machine. These can run on multiple machines and will sweep continuously, starting new `wandb` runs in succession. 
The second option starts a `wandb` agent that will only start a single `wandb` run. This can also be run on multiple machines and each will terminate after the 
run is complete. The last option queues this script on a remote machine using the SLURM job-queuing system. You will need to adjust the SLURM script to suit your needs. 

