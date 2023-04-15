""" Main trainer class. """

import logging
import os
import shutil
import time
from pathlib import Path

# typing imports
from typing import Dict, List, Optional

import numpy as np
import torch
from huggingface_hub import Repository, create_repo
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import HubStrategy, has_length, speed_metrics
from transformers.utils import get_full_repo_name

from .config import TransformerSegmentationConfig

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: TransformerSegmentationConfig,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
        """

        self.hydra_config = hydra_config

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name

        super().__init__(**kwargs)

    def init_git_repo(self, at_init: bool = False) -> None:
        """
        Initializes a git repo in `self.args.hub_model_id`.
        Args:
            at_init (`bool`, *optional*, defaults to `False`):
                Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is
                `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped
                out.
        """
        if not self.is_world_process_zero():
            return
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id
        if "/" not in repo_name:
            repo_name = get_full_repo_name(
                repo_name, token=self.args.hub_token
            )

        # Make sure the repo exists.
        create_repo(
            repo_name,
            token=self.args.hub_token,
            private=self.args.hub_private_repo,
            exist_ok=True,
        )
        try:
            self.repo = Repository(
                self.args.output_dir,
                clone_from=repo_name,
                token=self.args.hub_token,
                revision=self.experiment_name,
            )
        except EnvironmentError:
            if self.args.overwrite_output_dir and at_init:
                # Try again after wiping output_dir
                shutil.rmtree(self.args.output_dir)
                self.repo = Repository(
                    self.args.output_dir,
                    clone_from=repo_name,
                    token=self.args.hub_token,
                    revision=self.experiment_name,
                )
            else:
                raise

        try:
            # the branch name should have been created already by the `create_repo` call
            self.repo.git_pull()
        except OSError:
            # if the repo is empty, the git_pull will fail
            pass

        # By default, ignore the checkpoint folders
        if (
            not os.path.exists(
                os.path.join(self.args.output_dir, ".gitignore")
            )
            and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS
        ):
            with open(
                os.path.join(self.args.output_dir, ".gitignore"),
                "w",
                encoding="utf-8",
            ) as writer:
                writer.writelines(["checkpoint-*/"])

        self.push_in_progress = None

        config_output_path = os.path.join(
            self.args.output_dir, f"hydra_config_{time.time()}.yaml"
        )
        OmegaConf.save(self.hydra_config, config_output_path)
