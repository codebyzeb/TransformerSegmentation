""" Utilities for setting up logging and reading configuration files"""

import os
import logging 
import torch 
import numpy as np
import random
import json

from configparser import ConfigParser

def set_seed(seed):
    """ Sets seed for reproducibility """
    if seed < 0: 
        logging.info('Skipping seed setting for reproducibility')
        logging.info('If you would like to set a seed, set seed to a positive value in config')
        return

    logging.info(f'Setting seed: {seed}')
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def setup_config(config_file_path):
    """ Reads in a config file using ConfigParser """
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(experiment_directory):
    """ Sets up logging functionality """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = os.path.join(experiment_directory, 'experiment.log')
    logging.basicConfig(
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(log_file_name),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f'Initializing experiment: {experiment_directory}')

def setup(config_file_path, run_id):
    """ Reads in config file, sets up logger and sets a seed to ensure reproducibility. """
    config = setup_config(config_file_path)

    # Create experiment directory
    exp_dir = os.path.join(config.get('EXPERIMENT', 'name', fallback=""), str(run_id))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    config['EXPERIMENT']['directory'] = exp_dir

    # Set up logger
    setup_logger(exp_dir)

    # Set up seed
    seed=config.getint('EXPERIMENT', 'seed', fallback=-1)
    config['EXPERIMENT']['seed'] = str(seed)
    set_seed(seed)

    return config
