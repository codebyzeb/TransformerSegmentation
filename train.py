"""Train a GPT2 model on the Phonemized EnglishNA CHILDES dataset."""

import logging
import os

# config-related imports
import hydra

# training pipeline imports
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import TrainingArguments

# wandb for logging metrics
import wandb
from src.config import TransformerSegmentationConfig
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


@hydra.main(version_base=None, config_path="conf", config_name="br")
def main(cfg: TransformerSegmentationConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

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
    model = load_model(cfg, len(tokenizer.get_vocab()))

    # Preprocess data
    logger.info("Preprocessing data")

    data_preprocessor = DataPreprocessor(cfg.data_preprocessing, tokenizer)

    num_rows = dataset["validation"].num_rows
    segment_eval_sentences = dataset["validation"].select(
        range(num_rows - 3000, num_rows)
    )["text"]
    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        # num_proc=64,
        remove_columns=["text"],
    )

    # Create labels
    processed_dataset = processed_dataset.map(
        lambda x: {"labels": x["input_ids"]}
    )

    # Remove all items that are shorter than the minimum length
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"])
        == cfg.data_preprocessing.max_input_length
    )

    # Setting up wandb
    if cfg.experiment.dry_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        os.environ["WANDB_ENTITY"] = "zeb"
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    # Set up training arguments
    # TODO: If we are using wandb sweeps, note that we will need to think about how we store/
    # initialize the name of the current experiment so that it doesn't interfere with the name
    # of other experiments, and also so that we can store checkpoints of that run on HF hub;
    # alternatively maybe we use ray tune which is natively supported by Trainer

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
        eval_steps=cfg.trainer.max_training_steps
        // 10,  # evaluate every 10% of training
        save_steps=cfg.trainer.max_training_steps
        // 10,  # checkpoint every 10% of training
        logging_steps=cfg.trainer.max_training_steps
        // 100,  # log every 1% of training
        run_name=cfg.experiment.name,
        report_to="wandb"
        if not cfg.experiment.dry_run
        else None,  # wandb deactivated for dry runs
        save_strategy="no" if cfg.experiment.dry_run else "steps",
        hub_strategy="every_save",
        push_to_hub=not cfg.experiment.dry_run,
        hub_model_id=f"transformersegmentation/{cfg.experiment.group}-{cfg.model.name}-model"
        if not cfg.experiment.dry_run
        else None,
        hub_token=os.environ["HF_WRITE_TOKEN"]
        if not cfg.experiment.dry_run
        else None,
        remove_unused_columns=True,
        label_names=["input_ids"],
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        segment_eval_sentences=segment_eval_sentences,
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train(resume_from_checkpoint=cfg.model.resume_checkpoint_path)


if __name__ == "__main__":
    main()
