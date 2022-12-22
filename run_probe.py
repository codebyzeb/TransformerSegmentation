__author__ = 'Zeb Goriely'
""" Entry point for launching probe """

import argparse
import os
import random
import logging
import wandb
import yaml

from src.segmentation.probing import ProbeTrainer

def get_args():

    # Unknown CLI arguments are automatically passed to wandb 
    parser = argparse.ArgumentParser(description="Parses config files passed in via CLI. Arguments passed to the CLI override those in the config file.")
    parser.add_argument("Path", metavar='path', type=str, help='path to the wandb run directory')
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
    setup_logging()

    run_dir = args.Path
    config_file = os.path.join(run_dir, 'config.yaml')
    checkpoint_file = os.path.join(run_dir, 'latest-checkpoint.pt')
    train_data = 'data/Eng-NA/valid.txt'
    test_data = 'data/Eng-NA/test.txt'

    # Create probe trainer
    probe_trainer = ProbeTrainer(config_file, checkpoint_file, train_data, test_data)

    # Train probe
    probe_trainer()

if __name__ == '__main__':
    main()
