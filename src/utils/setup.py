""" Utilities for setting up experiments"""

from typing import Optional
import datasets
import logging
import numpy as np
import random
import os
import torch
import wandb

from src import config
from omegaconf import OmegaConf

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

def check_and_set_environment_variables(cfg: config.TransformerSegmentationConfig) -> None:
    """ Checks huggingface tokens exist and sets up wandb environment variables """

    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables. Check .env file and source it if necessary."

    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        if cfg.experiment.resume_run_id is not None:
            raise RuntimeError("resume_run_id is set but offline_run is True. Ignoring resume_run_id.")
    else:
        assert (
            "WANDB_ENTITY" in os.environ
        ), "WANDB_ENTITY needs to be set as an environment variable if not running in offline mode. Check .env file and source it if necessary."
        if cfg.experiment.resume_checkpoint_path and cfg.experiment.resume_run_id is None:
            raise RuntimeError("resume_run_id must be set if resume_checkpoint_path is set")

     # Disable parallelism in tokenizers to avoid issues with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_config(cfg: config.TransformerSegmentationConfig) -> None:
    """ Infer missing config values (checkpoint path and/or experiment name) and check if keys are missing """

    # It is possible to infer the name if resume_run_id is set or if resume_checkpoint_path is set. Otherwise, a random name is generated.
    if cfg.experiment.resume_run_id:
        # Case when resume_run_id provided but not the experiment name
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if "name" not in cfg.experiment:
            api = wandb.Api()
            run = api.run(f"{wandb_entity}/{cfg.experiment.group}/{cfg.experiment.resume_run_id}")
            cfg.experiment.name = run.name
            logger.info(f"experiment.name not set, loaded {cfg.experiment.name} from resume_run_id {cfg.experiment.resume_run_id} on wandb.")
        # Case when resume_run_id provided but not the checkpoint path
        if not cfg.experiment.resume_checkpoint_path:
            checkpoint_paths = [dir for dir in os.listdir(f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}") if dir.startswith("checkpoint")]
            if len(checkpoint_paths) > 0:
                checkpoint_numbers = [int(path.split("-")[-1]) for path in checkpoint_paths]
                checkpoint_numbers.sort()
                cfg.experiment.resume_checkpoint_path = f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}/checkpoint-{checkpoint_numbers[-1]}"
                logger.info(f"resume_checkpoint_path not set, loaded {cfg.experiment.resume_checkpoint_path} from latest checkpoint.")
            else:
                raise RuntimeError(f"resume_run_id set but no checkpoints found in the run directory checkpoints/{cfg.experiment.group}/{cfg.experiment.name}. Please specify resume_checkpoint_path.")
    if "name" not in cfg.experiment:
        # Case when checkpoint_path is provided but not the experiment name
        if cfg.experiment.resume_checkpoint_path is not None:
            cfg.experiment.name = cfg.experiment.resume_checkpoint_path.split("/")[-2]
            logger.warning(f"experiment.name not set, infering {cfg.experiment.name} from resume_checkpoint_path.")
        # Case when neither resume_run_id nor resume_checkpoint_path is provided. Generate a random name.
        else:
            cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(9999, (1,)).item()).zfill(4)}"
            if not cfg.experiment.offline_run:
                wandb_entity = os.environ.get("WANDB_ENTITY")
                api = wandb.Api()
                runs = api.runs(f"{wandb_entity}/{cfg.experiment.group}")
                while any(run.name == cfg.experiment.name for run in runs):
                    cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(9999, (1,)).item()).zfill(4)}"
            logger.warning(f"experiment.name not set, generated random name {cfg.experiment.name}")

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")
    if cfg.data_preprocessing.join_utts not in ["dynamic", "static", None, "None"]:
        raise RuntimeError(f"Invalid value for join_utts: {cfg.data_preprocessing.join_utts}. Must be one of 'dynamic', 'static', or None.")
    if cfg.data_preprocessing.join_utts == "None":
        cfg.data_preprocessing.join_utts = None
    if cfg.data_preprocessing.subsample_type not in ["examples", "words", "tokens", None]:
        raise RuntimeError(f"Invalid value for subsample_type: {cfg.data_preprocessing.subsample_type}. Must be one of 'examples', 'words', or 'tokens'.")
    if cfg.experiment.evaluate_babyslm and "English" not in cfg.dataset.subconfig:
        raise RuntimeError("evaluate_babyslm is only supported for the English dataset.")

def load_dataset(cfg : config.DatasetParams) -> datasets.Dataset:
    """ Loads dataset from config """
    text_column = cfg.text_column
    load_columns = [text_column, "target_child_age"]
    features = datasets.Features({text_column: datasets.Value("string"), "target_child_age": datasets.Value("float")})
    dataset = datasets.load_dataset(
        cfg.name,
        cfg.subconfig,
        token=os.environ["HF_READ_TOKEN"],
        column_names=load_columns,
        features=features,
    )

    # Drop rows where target_child_age is none or is larger than the max_age
    if cfg.max_age is not None:
        dataset = dataset.filter(lambda x: x["target_child_age"] is not None and x["target_child_age"] <= cfg.max_age, num_proc=(64 if torch.cuda.is_available() else 1))

    # Rename target column to "text"
    dataset = dataset.rename_column(text_column, "text")
    return dataset

def setup_wandb(cfg : config.ExperimentParams) -> None:
    """ Setup wandb for logging """
    wandb_entity = os.environ["WANDB_ENTITY"]
    os.environ["WANDB_PROJECT"] = cfg.group
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.resume_checkpoint_path:
        os.environ["WANDB_RUN_ID"] = cfg.resume_run_id
        os.environ["WANDB_RESUME"] = "allow"
    wandb.init(
        entity=wandb_entity,
        project=cfg.group,
        name=cfg.name,
        config=wandb.config,
        resume="allow",
        id=cfg.resume_run_id,
    )

def subsample_dataset(dataset : datasets.Dataset, num_examples : int, subsample_type : Optional[str]) -> datasets.Dataset:
    """ Subsample the dataset """
    if subsample_type is None or subsample_type == "examples":
        return dataset.shuffle().select(range(num_examples))
    elif subsample_type == "words":
        sampled_dataset = dataset.shuffle()
        cumulative_word_count = 0
        for i, example in enumerate(list(sampled_dataset["word_starts"])):
            cumulative_word_count += example.count(1)
            if cumulative_word_count >= num_examples:
                return sampled_dataset.select(range(i))
        raise RuntimeError(f"Subsample count {num_examples} exceeds total word count in dataset")
    elif subsample_type == "tokens":
        sampled_dataset = dataset.shuffle()
        cumulative_token_count = 0
        for i, example in enumerate(list(sampled_dataset["input_ids"])):
            cumulative_token_count += len(example)
            if cumulative_token_count >= num_examples:
                return sampled_dataset.select(range(i))
        raise RuntimeError(f"Subsample count {num_examples} exceeds total token count in dataset")
    else:
        raise RuntimeError(f'Invalid value "{subsample_type}" for subsample_type. Must be one of "examples", "tokens" or "words".')