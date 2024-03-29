"""Train a GPT2 model on the Phonemized EnglishNA CHILDES dataset."""

import logging
import os
import torch

# config-related imports
import hydra

# training pipeline imports
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers.trainer import TrainingArguments
from transformers.data import DataCollatorForLanguageModeling

# wandb for logging metrics
import wandb
from src.config import TransformerSegmentationConfig
from src.datacollator import CustomDataCollatorForLanguageModeling
from src.models import load_model
from src.preprocessing import DataPreprocessor
from src.tokenizer import load_tokenizer
from src.trainer import CustomTrainer
from src.utils.setup import set_seed

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=TransformerSegmentationConfig)

# A logger for this file
logger = logging.getLogger(__name__)

DRY_RUN_SUBSAMPLE_FACTOR = 10 // (10 if torch.cuda.device_count() > 1 else 1)
DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TransformerSegmentationConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    if 'name' not in cfg.experiment.__dict__:
        # Set to dataset subconfig with random 5 digit number
        cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(10000, (1,)).item()).zfill(5)}"
        logger.warning(f"experiment.name not set, using {cfg.experiment.name}")

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")
    if cfg.data_preprocessing.join_utts not in ["dynamic", "static", None]:
        raise RuntimeError(f"Invalid value for join_utts: {cfg.data_preprocessing.join_utts}. Must be one of 'dynamic', 'static', or None.")

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    if cfg.experiment.dry_run:
        logger.info("Running in dry run mode -- overriding config with values: ")
        logger.info(f"\t max_training_steps: {DRY_RUN_TRAIN_STEPS}")
        logger.info(f"\t num_warmup_steps: {DRY_RUN_WARMUP_STEPS}")
        cfg.trainer.max_training_steps = DRY_RUN_TRAIN_STEPS
        cfg.trainer.num_warmup_steps = DRY_RUN_WARMUP_STEPS

    # Loading dataset
    logger.info("Loading dataset")
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg, dataset)

    # Load model
    logger.info("Initializing model")
    model = load_model(cfg, tokenizer)

    # Get a sample of the validation set for evaluation
    num_rows = dataset["validation"].num_rows
    segment_eval_sentences = dataset["validation"].select(range(num_rows - 3000, num_rows))["text"]

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DataPreprocessor(cfg.data_preprocessing, tokenizer)

    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        # num_proc=64,
        remove_columns=["text"],
    )

    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["validation"]
    if cfg.experiment.dry_run:
        logger.info(f"Running in dry run mode -- subsampling dataset by {DRY_RUN_SUBSAMPLE_FACTOR}x")
        train_dataset = train_dataset.select(range(0, train_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR))
        eval_dataset = eval_dataset.select(range(0, eval_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR))

    # Set up custom data collator which joins examples to fill the context size
    if cfg.data_preprocessing.join_utts == 'dynamic':
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer, max_seq_length=cfg.data_preprocessing.max_input_length, mlm=False)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Setting up wandb
    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        os.environ["WANDB_ENTITY"] = "zeb"
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        if cfg.experiment.resume_checkpoint_path:
            resume_run_id = cfg.experiment.resume_run_id
            if resume_run_id is None:
                raise RuntimeError("resume_run_id must be set if resume_checkpoint_path is set")
            os.environ["WANDB_RUN_ID"] = resume_run_id
            os.environ["WANDB_RESUME"] = "allow"
        wandb.init(
            entity="zeb",
            project=cfg.experiment.group,
            name=cfg.experiment.name,
            config=wandb.config,
            resume="allow",
            id=cfg.experiment.resume_run_id,
        )
        # Report number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of parameters: {num_params}")
        wandb.config.update({"num_params": num_params})

        # Report length of data
        logger.info(f"Number of training examples: {train_dataset.num_rows}")
        logger.info(f"Number of validation examples: {eval_dataset.num_rows}")
        wandb.config.update({"num_train_examples": train_dataset.num_rows})
        wandb.config.update({"num_val_examples": eval_dataset.num_rows})

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy="steps",
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        eval_steps=cfg.trainer.max_training_steps // (2 if cfg.experiment.dry_run else 50),  # evaluate every 2% of training
        save_steps=cfg.trainer.max_training_steps // (2 if cfg.experiment.dry_run else 10),  # checkpoint every 10% of training
        logging_steps=cfg.trainer.max_training_steps // 100,  # log every 1% of training
        run_name=cfg.experiment.name,
        report_to="wandb" if not cfg.experiment.offline_run else None,  # wandb deactivated for dry runs
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=not cfg.experiment.offline_run,
        hub_model_id=f"transformersegmentation/{cfg.experiment.group}-{cfg.model.name}-model" if not cfg.experiment.offline_run else None,
        hub_token=os.environ["HF_WRITE_TOKEN"] if not cfg.experiment.offline_run else None,
        remove_unused_columns=True,
        label_names=["input_ids"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,  # smaller perplexity is better
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        segment_eval_sentences=segment_eval_sentences,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

    # Evaluate best model
    trainer.stride_evaluation = 16
    trainer.evaluate(metric_key_prefix="eval_best")


if __name__ == "__main__":
    main()
