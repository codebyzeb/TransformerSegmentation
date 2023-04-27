import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import TransformerSegmentationConfig
from .gpt2 import *
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from ..preprocessing import create_phoneme_map

logger = logging.getLogger(__name__)


def load_model(
    cfg: TransformerSegmentationConfig, tokenizer, 
) -> PreTrainedModel:
    """Loads the model from the config file

    Args:
        cfg (TransformerSegmentationConfig): hydra config object
        tokenizer (PreTrainedTokenizer): tokenizer object
    """

    remove_keys = ["name", "load_from_checkpoint", "checkpoint_path"]
    model_kwargs = {
        key: val
        for key, val in cfg.model.items()
        if key not in remove_keys and val is not None
    }

    model_kwargs["vocab_size"] = tokenizer.vocab_size

    if cfg.model.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.model.name](**model_kwargs)
        if cfg.model.name == 'gpt2_feature_model':
            phoneme_map = create_phoneme_map(tokenizer, cfg.dataset.subconfig)
            model = MODEL_REGISTRY[cfg.model.name](config, phoneme_map)
        else:
            model = MODEL_REGISTRY[cfg.model.name](config)
    else:
        raise ValueError(f"Model {cfg.model.name} not found in registry")

    return model
