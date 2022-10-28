__author__ = 'Zeb Goriely'
""" Entry point for launching phoneme transformer """

import argparse
import random
import os
import signal
import shutil

from src.utils import setup
from src.phone_transformer import PhoneTransformer

parser = argparse.ArgumentParser(description="Parses config files passed in via CLI")
parser.add_argument("Path", metavar='path', type=str, help='path to the config file')
parser.add_argument('--run_id', type=str, help="""Unique identifier for the run of the model""")
args = parser.parse_args()

# ENTRY POINT 

def main():

    # Setting up an id for the specific run
    if args.run_id is None: 
        run_id = str(random.randint(1, 1e9))
    else: 
        run_id = args.run_id

    # Possibly reading in a runfile (only exists if we are resuming training)
    run_file_path = f"tmp/{run_id}.runfile"
    if os.path.exists(run_file_path): 
        # we must be resuming a run - reading in relevant information
        with open(run_file_path, "r") as f: 
            resume_num_epochs = int(f.readline())
    else: 
        # Beginining training - make sure config file isn't modified before training is resumed
        shutil.copyfile(args.Path, f'tmp/{run_id}.ini')
        resume_num_epochs = 0

    # Setting up logging, config read in and seed setting
    config = setup(args.Path, run_id, resume_num_epochs)
    
    # Initializing problyglot with configuration and options
    phonetransformer = PhoneTransformer(config, resume_num_epochs)

    # setting up timeout handler - called if the program receives a SIGINT either from the user
    # or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, phonetransformer.timeout_handler)

    # launching training or eval script
    phonetransformer()

    if os.path.exists(run_file_path):
        os.remove(run_file_path)

if __name__ == '__main__':
    main()
