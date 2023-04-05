""" Utilities for setting up experiments"""

import logging
import random

import numpy as np
import torch

# A logger for this file
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility"""
    if seed < 0:
        logger.warning("Skipping seed setting for reproducibility")
        logger.warning(
            "If you would like to set a seed, set seed to a positive value in config"
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)
