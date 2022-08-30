import torch

from .setup import setup, set_seed

# global device (used unless explicitly overriden)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()