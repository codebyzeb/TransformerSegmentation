# TransformerSegmentation
Can a character-based transformer learn word boundaries?

## Installation

To get setup create a hugging face account and ask @codebyzeb to add you to the group's private hugging face hub. The hub is where we keep data, tokenization, model and other artifacts. During training, we pull in these values directly from the hub (and occasionally also push progamatically to the hub). 

In order to interact with the hub, you need to generate read and write [access tokens](https://huggingface.co/docs/hub/security-tokens) from your hugging face account. Once generated, store these values as environment variables in a local `.env` file with the names `HF_READ_TOKEN`, and `HF_WRITE_TOKEN`.

If you plan to use Weights and Biases for logging (i.e. not running in offline mode) you will also need to specify your wandb project to `.env` with the name `WANDB_ENTITY`. 

Before running the code, make sure to run the setup script `./setup.sh`. This script sets up the requirements imports as well as git hooks for automatic code formatting. Additionally, this script makes sure you are logged into wandb and huggingface.

## Hyperparameter Tuning

Hyperparameter tuning was conducted using a Weights & Biases sweep. To create a sweep, run the following:

```
wandb sweep --project PROJECT_NAME configs/sweep.yaml
```

This will create a new sweep in Weights & Biases with an ID and a URL. You can either run the agent by either:
1. Running an agent directly using wandb: `wandb agent USER_ID/PROJECT_NAME/SWEEP_ID`
2. Run the script provided: `sh scripts/run_sweep_agent USER_ID/PROJECT_NAME/SWEEP_ID`
3. Queue a SLURM job to run the agent on a different node: `sbatch scripts/slurm_submit_sweep_agent.wilkes3 USER_ID/PROJECT_NAME/SWEEP_ID`.

The first option starts a `wandb` agent on your machine. These can run on multiple machines and will sweep continuosly, starting new `wandb` runs in succession. 
The second option starts a `wandb` agent that will only start a single `wandb` run. This can also be run on multiple machines and each will terminate after the 
run is complete. The last option queues this script on a remote machine using the SLURM job-queuing system. You will need to adjust the SLURM script to suit your needs. 

