__author__ = 'Zeb Goriely'
""" Entry point for launching phoneme transformer """

import argparse
import random
import logging
import numpy as np
import os
import signal
import shutil
import torch
import wandb

from src.phone_transformer import PhoneTransformer

parser = argparse.ArgumentParser(description="Parses config files passed in via CLI")
parser.add_argument("Path", metavar='path', type=str, help='path to the config file')
parser.add_argument('--run_id', type=str, help="""Unique identifier for the run of the model""")
args = parser.parse_args()

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

    # Setting up an id for the specific run
    if args.run_id is None: 
        run_id = str(random.randint(1, 1e9))
    else: 
        run_id = args.run_id

    setup_logging()

    # Set up WandB
    wandb.init(
        project='tmp',
        entity='zeb',
        config=args.Path,
        id=run_id,
        resume='allow'
    )
    logging.info(f'{"Resuming" if wandb.run.resumed else "Starting"} run with id: {run_id}')
    logging.info('Using the following configuration:')
    logging.info(wandb.config)

    # Initializing training script with configuration and options
    phonetransformer = PhoneTransformer()

    # setting up timeout handler - called if the program receives a SIGINT either from the user
    # or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, phonetransformer.timeout_handler)

    # launching training or eval script
    phonetransformer()

if __name__ == '__main__':
    main()
