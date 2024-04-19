""" Main trainer class. """

import logging
import math
import os
import shutil
import time
from pathlib import Path

# typing imports
from typing import List, Optional

import torch
from huggingface_hub import Repository, create_repo
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.trainer import Trainer
from transformers.trainer_utils import HubStrategy, seed_worker, speed_metrics
from transformers.utils import get_full_repo_name, is_datasets_available

from .config import TransformerSegmentationConfig
from .datasampler import CustomBatchSampler
from .segmentation.segment import GPT2FeaturesSegmenter, GPT2Segmenter

SEGMENTER_MAP = {
    "GPT2LMHeadModel": GPT2Segmenter,
    "GPT2FeatureModel": GPT2FeaturesSegmenter,
}

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: TransformerSegmentationConfig,
        segment_eval_sentences: List[str] = None,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
            * segment_eval_sentences: (List[str]): The sentences to evaluate segmentation on.
        """

        self.hydra_config = hydra_config
        self.max_seq_length = hydra_config.data_preprocessing.max_input_length
        self.segment_eval_sentences = segment_eval_sentences

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name

        self.stride_evaluation = None

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
            repo_name = get_full_repo_name(repo_name, token=self.args.hub_token)

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
        if not os.path.exists(os.path.join(self.args.output_dir, ".gitignore")) and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS:
            with open(
                os.path.join(self.args.output_dir, ".gitignore"),
                "w",
                encoding="utf-8",
            ) as writer:
                writer.writelines(["checkpoint-*/"])

        self.push_in_progress = None

        config_output_path = os.path.join(self.args.output_dir, f"hydra_config_{time.time()}.yaml")
        OmegaConf.save(self.hydra_config, config_output_path)

    def get_train_dataloader(self) -> DataLoader:
        """
        Overriden to use our custom batch sampler.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        train_sampler = self._get_train_sampler()

        if self.hydra_config.data_preprocessing.join_utts == "dynamic":
            batch_sampler = CustomBatchSampler(
                train_sampler,
                batch_size=self._train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                max_seq_length=self.max_seq_length,
            )
        else:
            batch_sampler = BatchSampler(train_sampler, batch_size=self._train_batch_size, drop_last=self.args.dataloader_drop_last)

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Overriden to use our custom batch sampler.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        eval_sampler = self._get_eval_sampler(eval_dataset)

        if self.hydra_config.data_preprocessing.join_utts == "dynamic":
            batch_sampler = CustomBatchSampler(
                eval_sampler,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                max_seq_length=self.max_seq_length,
            )
        else:
            batch_sampler = BatchSampler(eval_sampler, batch_size=self.args.eval_batch_size, drop_last=self.args.dataloader_drop_last)

        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Overriden to use our custom batch sampler.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        test_sampler = self._get_eval_sampler(test_dataset)

        if self.hydra_config.data_preprocessing.join_utts == "dynamic":
            batch_sampler = CustomBatchSampler(
                test_sampler,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                max_seq_length=self.max_seq_length,
            )
        else:
            batch_sampler = BatchSampler(test_sampler, batch_size=self.args.eval_batch_size, drop_last=self.args.dataloader_drop_last)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    # Override evaluate to do word segmentation
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.stride_evaluation is not None:
            # Create longs vector with all input ids and labels in eval_dataset
            long_input_ids = []
            for batch in eval_dataloader:
                long_input_ids.extend(batch["input_ids"].flatten().tolist())
            long_input_ids = torch.tensor(long_input_ids).to(self.args.device)
            long_target_ids = []
            for batch in eval_dataloader:
                long_target_ids.extend(batch["labels"].flatten().tolist())
            long_target_ids = torch.tensor(long_target_ids).to(self.args.device)

            max_length = self.max_seq_length
            stride = self.stride_evaluation
            input_id_length = len(long_input_ids)
            logger.info("Evaluating perplexity with stride %d and max length %d" % (stride, max_length))

            # Batch the input ids and labels into sequences of length max_length by shifting by stride
            input_ids = []
            target_ids = []
            prev_end_loc = 0
            batch_size = self.args.eval_batch_size
            for begin_loc in range(0, len(long_input_ids), stride):
                end_loc = min(begin_loc + max_length, len(long_input_ids))
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                inputs = long_input_ids[begin_loc:end_loc].to(self.args.device)
                targets = long_target_ids[begin_loc:end_loc].to(self.args.device).clone()
                targets[:-trg_len] = -100
                input_ids.append(inputs)
                target_ids.append(targets)
                prev_end_loc = end_loc
                if end_loc == input_id_length:
                    # Ensure final set of input_ids is the same length as the rest
                    input_ids[-1] = torch.cat((input_ids[-1], torch.zeros(stride - trg_len, dtype=torch.long, device=self.args.device)))
                    target_ids[-1] = torch.cat(
                        (target_ids[-1], torch.ones(stride - trg_len, dtype=torch.long, device=self.args.device) * -100)
                    )
                    break

            # Ensure divisible by batch_size
            while len(input_ids) % batch_size != 0:
                input_ids.append(torch.zeros_like(input_ids[0]))
                # Set to -100 for targets
                target_ids.append(torch.ones_like(target_ids[0]) * -100)

            # Stack into batches of batch_size
            input_ids = torch.stack(input_ids)
            target_ids = torch.stack(target_ids)
            input_ids = input_ids.view(-1, batch_size, max_length)
            target_ids = target_ids.view(-1, batch_size, max_length)
            seq_len = input_ids.size(0)

            nlls = []
            for i in tqdm(range(seq_len)):
                with torch.no_grad():
                    outputs = self.model(input_ids[i], labels=target_ids[i])
                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

            nlls = [nll for nll in nlls if not torch.isnan(nll)]
            ppl = torch.exp(torch.stack(nlls).mean())
            bpc = torch.stack(nlls).mean() / math.log(2)
            output.metrics[f"{metric_key_prefix}_stride_perplexity"] = ppl
            output.metrics[f"{metric_key_prefix}_stride_bpc"] = bpc

        # Get perplexity on evaluation set
        eval_loss = output.metrics[f"{metric_key_prefix}_loss"]
        perplexity = math.exp(eval_loss)
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        # Get bits per character on evaluation set
        eval_loss = output.metrics[f"{metric_key_prefix}_loss"]
        bits_per_character = eval_loss / math.log(2)
        output.metrics[f"{metric_key_prefix}_bpc"] = bits_per_character

        # Evaluate word segmentation on the segmentation evaluation sentences
        if self.segment_eval_sentences:
            model_class = self.model.__class__.__name__
            if model_class in SEGMENTER_MAP:
                segmenter = SEGMENTER_MAP[model_class](self.model, self.tokenizer, self.segment_eval_sentences)
                best_cutoffs_type = {}
                best_cutoffs_boundary = {}
                for measure in tqdm(segmenter.measures, desc="Evaluating segmentation measures"):
                    spike_seg_metrics = segmenter.evaluate_spike_segmentation(measure)
                    output.metrics[f"{metric_key_prefix}_spike_seg_type_fscore_{measure}"] = spike_seg_metrics["type_fscore"]
                    output.metrics[f"{metric_key_prefix}_spike_seg_boundary_fscore_{measure}"] = spike_seg_metrics["boundary_noedge_fscore"]

                    best_cutoffs_type[measure], type_fscore = segmenter.find_best_cutoff(measure, 'type_fscore')
                    output.metrics[f"{metric_key_prefix}_absolute_seg_type_fscore_{measure}"] = type_fscore
                    
                    best_cutoffs_boundary[measure], boundary_fscore = segmenter.find_best_cutoff(measure, 'boundary_noedge_fscore')
                    output.metrics[f"{metric_key_prefix}_absolute_seg_boundary_fscore_{measure}"] = boundary_fscore

                # Add majority vote measures based on type fscore votes
                segmenter.add_majority_vote(best_cutoffs_type)
                output.metrics[f"{metric_key_prefix}_spike_seg_type_fscore_Majority Vote Cutoff"] = segmenter.evaluate_spike_segmentation('Majority Vote Cutoff')["type_fscore"]
                output.metrics[f"{metric_key_prefix}_spike_seg_type_fscore_Majority Vote Spike"] = segmenter.evaluate_spike_segmentation('Majority Vote Spike')["type_fscore"]
                output.metrics[f"{metric_key_prefix}_absolute_seg_type_fscore_Majority Vote Cutoff"] = segmenter.evaluate_cutoff_segmentation('Majority Vote Cutoff', 0.5)["type_fscore"]
                output.metrics[f"{metric_key_prefix}_absolute_seg_type_fscore_Majority Vote Spike"] = segmenter.evaluate_cutoff_segmentation('Majority Vote Spike', 0.5)["type_fscore"]

                # Add majority vote measures based on boundary fscore votes
                segmenter.add_majority_vote(best_cutoffs_boundary)
                output.metrics[f"{metric_key_prefix}_spike_seg_boundary_fscore_Majority Vote Cutoff"] = segmenter.evaluate_spike_segmentation('Majority Vote Cutoff')["boundary_noedge_fscore"]
                output.metrics[f"{metric_key_prefix}_spike_seg_boundary_fscore_Majority Vote Spike"] = segmenter.evaluate_spike_segmentation('Majority Vote Spike')["boundary_noedge_fscore"]
                output.metrics[f"{metric_key_prefix}_absolute_seg_boundary_fscore_Majority Vote Cutoff"] = segmenter.evaluate_cutoff_segmentation('Majority Vote Cutoff', 0.5)["boundary_noedge_fscore"]
                output.metrics[f"{metric_key_prefix}_absolute_seg_boundary_fscore_Majority Vote Spike"] = segmenter.evaluate_cutoff_segmentation('Majority Vote Spike', 0.5)["boundary_noedge_fscore"]

            else:
                logging.warning(f"No segmenter available for model class {model_class}, skipping segmentation evaluation")

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
