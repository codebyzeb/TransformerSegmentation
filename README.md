# TransformerSegmentation
Can a character-based transformer learn word boundaries?

## Installation

To run the project, first install the required packages in requirements.txt by creating a conda environment or otherwise:

```
conda create --name transformer_seg python=3.8 pip
conda activate transformer_seg
pip install -r requirements.txt
```

## Usage



## Continuous Training

This project was developed using access to remote GPU nodes. Jobs could be queued onto these nodes using SLURM, with a cut-off of 12 hours.
In order to train for longer than 12 hours, use the `--requeue-after NUM_HOURS` option when calling `scripts/run_project.sh`.  
When `NUM_HOURS` hours have passed, the training script will automatically save the current state using `wandb` and exit with code 124. `run_project.sh` will 
then detect this exit code and launch a new SLURM job. It's an ugly solution but it works for our purposes.

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

Note that the sweep config launches the training process, with a requeue cutoff of 11 hours (see **Continuous Training** above). If you can run your training script continuously,
you should remove this option before creating a sweep as it will try to launch a new SLURM job after 11 hours otherwise. 

