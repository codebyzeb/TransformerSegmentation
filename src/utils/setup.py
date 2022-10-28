""" Utilities for setting up logging and reading configuration files"""

import os
import logging 
import torch 
import numpy as np
import random
import json
import wandb

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

def setup_logger(config_file_path):
    """ Sets up logging functionality """
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    experiment_directory = os.path.dirname(os.path.join(os.getcwd(), config_file_path)) 

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

def setup_wandb(config, run_id, resume_training):
    """
    Sets up logging and model experimentation using weights & biases 
    """
    if config.getboolean('EXPERIMENT', 'use_wandb', fallback=True):
        dict_config = json.loads(json.dumps(config._sections))
        wandb.init(project=config.get('EXPERIMENT', 'name'),
                   entity='zeb',
                   config=dict_config,
                   id=run_id,
                   resume='must' if resume_training else None
                   )
        if resume_training:
            logging.info(f'Resuming run with id: {run_id}')
        else: 
            logging.info(f'Starting run with id: {run_id}')

    logging.info('Using the following configuration:')
    defaults = config.defaults()
    if defaults:
        logging.info('DEFAULTS')
        for key, value in defaults:
            logging.info(f'   {key} : {value}')
    for section in config.sections():
        logging.info(section)
        for key, value in config.items(section):
            logging.info(f'   {key} : {value}')

def setup(config_file_path, run_id, resume_num_epochs):
    """ Reads in config file, sets up logger and sets a seed to ensure reproducibility. """
    config = setup_config(config_file_path)

    # we are resuming training if resume_num_epochs is greater than 0
    resume_training = resume_num_epochs > 0 

    # Set up logger and wandb
    setup_logger(config_file_path)
    setup_wandb(config, run_id, resume_training)

    # Set up seed
    seed=config.getint("EXPERIMENT", "seed", fallback=-1)

    # shifting over the random seed by resume_num_epochs steps in order for the meta
    # dataset to not yield the same sentences as already seen by the model 
    # also added benefit that if the same job is run again we are likely to achieve the same  
    # result
    seed += resume_num_epochs
    
    config["EXPERIMENT"]["seed"] = str(seed)
    set_seed(seed)

    return config
