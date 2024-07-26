import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import ModelParams
from .gpt2 import *
from .llama import *
from .gptneox import *
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from ..preprocessing import create_phoneme_map

logger = logging.getLogger(__name__)


def load_model(
    cfg: ModelParams, tokenizer, 
) -> PreTrainedModel:
    """Loads the model from the config file

    Args:
        cfg (TransformerSegmentationConfig): hydra config object
        tokenizer (PreTrainedTokenizer): tokenizer object
    """

    model_kwargs = dict(cfg.model_kwargs)

    #model_kwargs["vocab_size"] = tokenizer.vocab_size
    model_kwargs["bos_token_id"] = tokenizer.bos_token_id
    model_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if cfg.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.name](**model_kwargs)
        if config.name_or_path:
            model = MODEL_REGISTRY[cfg.name].from_pretrained(config.name_or_path)
            logger.info(f"Loaded model config from {config.name_or_path}")
        else:
            logging.info(f"Initialising model {cfg.name} with config {config} from scratch")
            if cfg.name == 'gpt2_feature_lm':
                phoneme_map = create_phoneme_map(tokenizer)
                model = MODEL_REGISTRY[cfg.name](config, phoneme_map)
            model = MODEL_REGISTRY[cfg.name](config)
    else:
        raise ValueError(f"Model {cfg.name} not found in registry")
    
    return model
