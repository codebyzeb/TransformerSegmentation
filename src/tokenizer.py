""" Tokenizer module """

import logging
import os

# typing imports
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import TransformerSegmentationConfig

logger = logging.getLogger(__name__)


def load_tokenizer(
    cfg: TransformerSegmentationConfig, dataset: Dataset
) -> PreTrainedTokenizer:
    """
    Sets up tokenizer for the model, based on tokenizer configurations

    Args:
        cfg (TransformerSegmentationConfig): hydra config object
        dataset (Dataset): instantiated dataset object
    """

    full_tokenizer_name = cfg.tokenizer.name
    org_name = full_tokenizer_name.split("/")[0]

    if org_name != "transformersegmentation":
        raise ValueError(
            "Tokenizer must be hosted on TransformerSegmentation. Please change the name in the config file."
        )

    # anything that's not name and vocab_size is an optional tokenizer kwarg
    remove_keys = ["name", "vocab_size"]
    tokenizer_kwargs = {
        key: val
        for key, val in cfg.tokenizer.items()
        if key not in remove_keys and val is not None
    }

    logging.info("Loading in tokenizer from hub")
    tokenizer = AutoTokenizer.from_pretrained(
        full_tokenizer_name,
        **tokenizer_kwargs,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    return tokenizer
