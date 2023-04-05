"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import MISSING


@dataclass
class ExperimentParams:
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to
    # analogous to 'project' in wandb
    group: str = MISSING

    # whether to run the experiment only locally
    dry_run: bool = False


@dataclass
class DatasetParams:
    # name of the dataset on huggingface
    name: str
    # subconfig i.e. full
    subconfig: str


@dataclass
class TokenizerParams:
    # data processing parameters
    name: str


@dataclass
class DataPreprocessingParams:
    # params for preprocessing the dataset (i.e. tokenization)
    max_input_length: int
    join_utts: bool


@dataclass
class ModelParams:
    # model parameters
    name: str

    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float
    dropout: float

    resume_checkpoint_path: Optional[str] = None


@dataclass
class TrainerParams:
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int


### Container for entire config ###


@dataclass
class TransformerSegmentationConfig:
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    trainer: TrainerParams
