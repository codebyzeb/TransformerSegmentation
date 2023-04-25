"""Train a GPT2 model on the Phonemized EnglishNA CHILDES dataset."""

import logging
import os

import evaluate

# config-related imports
import hydra
import numpy as np

# training pipeline imports
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import TrainingArguments

# wandb for logging metrics
import wandb
from src.config import TransformerSegmentationConfig
from src.models.gpt2 import GPT2Probe
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

    logger.info("Loading model")
    hub_model_name = f"transformersegmentation/{cfg.experiment.group}-{cfg.model.name}-model"
    model = GPT2Probe.from_pretrained(
        hub_model_name, revision=cfg.experiment.name
    )

    logger.info("Preprocessing data")
    cfg.data_preprocessing.join_utts = False  # Always false for probe
    data_preprocessor = DataPreprocessor(cfg.data_preprocessing, tokenizer)

    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        # num_proc=64,
        remove_columns=["text"],
    ).rename_column("word_starts", "labels")
    train_dataset = (
        processed_dataset["train"]
        .shuffle(seed=42)
        .select(range(min(10000, len(dataset["train"]))))
    )
    evaluate_dataset = (
        processed_dataset["validation"].shuffle(seed=42).select(range(1000))
    )

    # Setting up wandb
    if cfg.experiment.dry_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = (
            cfg.experiment.group + "-Segmentation-Probe"
        )
        os.environ["WANDB_ENTITY"] = "zeb"
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    # Setting up evaluation metrics
    seg_metrics = evaluate.load("transformersegmentation/segmentation_scores")

    def compute_metrics(eval_pred):
        logits, labels, inputs = eval_pred
        gold_sentences = []
        predict_sentences = []
        for i in range(logits.shape[0]):
            predict_boundaries = np.argmax(logits[i], axis=-1)
            gold_boundaries = labels[i]
            tokens = tokenizer.convert_ids_to_tokens(
                inputs[i], skip_special_tokens=True
            )
            predict_sentence = " ".join(
                [
                    "WORD_BOUNDARY " + token
                    if predict_boundaries[k]
                    else token
                    for k, token in enumerate(tokens)
                ]
            ).replace("UTT_BOUNDARY", "")
            gold_sentence = " ".join(
                [
                    "WORD_BOUNDARY " + token if gold_boundaries[k] else token
                    for k, token in enumerate(tokens)
                ]
            ).replace("UTT_BOUNDARY", "")
            predict_sentences.append(predict_sentence)
            gold_sentences.append(gold_sentence)

        return seg_metrics.compute(
            predictions=predict_sentences, references=gold_sentences
        )

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}-probe",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=15,
        seed=cfg.experiment.seed,
        run_name=cfg.experiment.name,
        report_to="wandb"
        if not cfg.experiment.dry_run
        else None,  # wandb deactivated for dry runs
        save_strategy="no" if cfg.experiment.dry_run else "epoch",
        push_to_hub=not cfg.experiment.dry_run,
        hub_model_id=f"transformersegmentation/{cfg.experiment.group}-Segmentation-Probe-{cfg.model.name}-model"
        if not cfg.experiment.dry_run
        else None,
        hub_token=os.environ["HF_WRITE_TOKEN"]
        if not cfg.experiment.dry_run
        else None,
        remove_unused_columns=True,
        include_inputs_for_metrics=True,
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluate_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()


# # Count 1s in labels
# import numpy as np
# num_pad = sum([np.sum(np.array(p) == 0) for p in evaluate_dataset['attention_mask']])
# total = sum([len(p) for p in evaluate_dataset['labels']]) - num_pad
# p_boundaries = np.sum(np.array(evaluate_dataset['labels']))/total
# p_no_boundaries = 1 - p_boundaries

# class RecallTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop('labels')
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         loss_fct = CrossEntropyLoss(weight=torch.tensor([1/p_no_boundaries, 1/p_boundaries], dtype=torch.float))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), torch.tensor(labels, dtype=torch.long).view(-1))
#         return (loss, outputs) if return_outputs else loss
