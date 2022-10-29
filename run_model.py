__author__ = 'Zeb Goriely'
""" Entry point for launching phoneme transformer """

import argparse
import random
import logging
import signal
import wandb
import yaml

from src.phone_transformer import PhoneTransformer

def get_args():

    # Unknown CLI arguments are automatically passed to wandb 
    parser = argparse.ArgumentParser(description="Parses config files passed in via CLI. Arguments passed to the CLI override those in the config file.")
    parser.add_argument("Path", metavar='path', type=str, help='path to the config file')
    parser.add_argument('--run_id', type=str, help="""Unique identifier for the run of the model""")
    parser.add_argument('--resubmit_after', type=float, default=None, help="""If set, will resubmit a new job to continue training if not complete after this many hours""")
    args, _ = parser.parse_known_args()
    return args

def setup_logging():
    """ Set up logger """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()]
    )

# ENTRY POINT 
def main():

    args = get_args()

    # Setting up an id for the specific run
    if args.run_id is None: 
        run_id = str(random.randint(1, 1e9))
    else: 
        run_id = args.run_id

    setup_logging()

    # Set up WandB
    with open(args.Path, 'r') as f:
        p = yaml.safe_load(f)
        experiment_name = p['experiment_name']['value']

    wandb.init(
        project=experiment_name,
        entity='zeb',
        config=args.Path,
        id=run_id,
        resume='allow'
    )
    wandb.config.update({"experiment_name" : wandb.run.project}, allow_val_change=True)
    logging.info(f'{"Resuming" if wandb.run.resumed else "Starting"} run with id: {run_id}')
    logging.info('Using the following configuration:')
    logging.info(wandb.config)

    # Initializing training script with configuration and options
    phonetransformer = PhoneTransformer(resubmit_after=args.resubmit_after)

    # launching training or eval script
    phonetransformer()

if __name__ == '__main__':
    main()
