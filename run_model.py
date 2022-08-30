__author__ = 'Zeb Goriely'
""" Entry point for launching phoneme transformer """

import argparse
import random

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

    # Setting up logging, config read in and seed setting
    config = setup(args.Path, run_id)
    
    # Initializing problyglot with configuration and options
    phonetransformer = PhoneTransformer(config)

    # launching training or eval script
    phonetransformer()

if __name__ == '__main__':
    main()
