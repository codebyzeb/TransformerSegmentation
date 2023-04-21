""" Tokenizer module """

import logging
import os

# typing imports
from datasets import Dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, GPT2TokenizerFast, PreTrainedTokenizer

from .config import TransformerSegmentationConfig

logger = logging.getLogger(__name__)


def create_tokenizer(
    cfg: TransformerSegmentationConfig, dataset: Dataset
) -> PreTrainedTokenizer:
    """
    Sets up custom tokenizer for the model, based on tokenizer configurations. The tokenizer simply splits on whitespace
    and removes the ;eword token.

    Args:
        cfg (TransformerSegmentationConfig): hydra config object
        dataset (Dataset): instantiated dataset object
    """

    tokenizer = Tokenizer(models.WordLevel(unk_token="UNK"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(";eword", "WORD_BOUNDARY "),
            normalizers.Replace("\n", "UTT_BOUNDARY"),
            normalizers.Strip(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["UNK", "PAD", "BOS", "EOS"]
    )
    tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer)

    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.bos_token = "BOS"
    wrapped_tokenizer.eos_token = "EOS"
    wrapped_tokenizer.pad_token = "PAD"
    wrapped_tokenizer.unk_token = "UNK"

    return wrapped_tokenizer


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
