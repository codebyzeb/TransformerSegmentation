"""Train a GPT2 model on the Phonemized EnglishNA CHILDES dataset."""

import logging
import os

# config-related imports
import hydra
import torch

# training pipeline imports
from datasets import Features, Value, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers.data import DataCollatorForLanguageModeling
from transformers.trainer import TrainingArguments

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
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables. Check .env file and source it if necessary."

    # Check and set up wandb environment variables depending on whether the run is offline or not
    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        wandb_entity = None
        if cfg.experiment.resume_run_id is not None:
            raise RuntimeError("resume_run_id is set but offline_run is True. Ignoring resume_run_id.")
    else:
        assert (
            "WANDB_ENTITY" in os.environ
        ), "WANDB_ENTITY needs to be set as an environment variable if not running in offline mode. Check .env file and source it if necessary."
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if cfg.experiment.resume_checkpoint_path and cfg.experiment.resume_run_id is None:
            raise RuntimeError("resume_run_id must be set if resume_checkpoint_path is set")

    # Disable parallelism in tokenizers to avoid issues with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # It is possible to infer the name if resume_run_id is set or if resume_checkpoint_path is set. Otherwise, a random name is generated.
    if cfg.experiment.resume_run_id:
        # Case when resume_run_id provided but not the experiment name
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
    if cfg.experiment.evaluate_babyslm and "English" not in cfg.dataset.subconfig:
        raise RuntimeError("evaluate_babyslm is only supported for the English dataset.")
    if cfg.data_preprocessing.join_utts == "None":
        cfg.data_preprocessing.join_utts = None

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
    # TODO: Custom data loading depending on if CHILDES or BR dataset
    logger.info("Loading dataset")
    load_columns = ["phonemized_utterance", "target_child_age"]
    features = Features({"phonemized_utterance": Value("string"), "target_child_age": Value("float")})
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        token=os.environ["HF_READ_TOKEN"],
        column_names=load_columns,
        features=features,
    )

    # Drop rows where target_child_age is none or is larger than the max_age
    if cfg.dataset.max_age is not None:
        dataset = dataset.filter(lambda x: x["target_child_age"] is not None and x["target_child_age"] <= cfg.dataset.max_age, num_proc=(64 if torch.cuda.is_available() else 1))

    # Rename "phonemized_utterance" to "text"
    dataset = dataset.rename_column("phonemized_utterance", "text")

    logging.info(f"Dataset loaded with {len(dataset['train'])} training examples")

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg, dataset)

    # Load model
    logger.info("Initializing model")
    model = load_model(cfg, tokenizer)

    # Report number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Get a sample of the validation set for evaluating segmentation. We do this before preprocessing because
    # preprocessing removes word boundaries, which we need as labels for evaluation.
    segment_eval_sentences = dataset["valid"]["text"] if cfg.experiment.evaluate_segmentation else None

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DataPreprocessor(cfg.data_preprocessing, tokenizer)

    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=(64 if torch.cuda.is_available() else 1),
        remove_columns=["text", "target_child_age"],
    )

    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["valid"]
    if cfg.dataset.num_examples is not None:
        logger.info(f"Subsampling training dataset to {cfg.dataset.num_examples} examples, evenly spaced across the dataset.")
        #train_dataset = train_dataset.select(range(0, train_dataset.num_rows, train_dataset.num_rows // cfg.dataset.num_examples))
        # Shuffle then take the first num_examples. Note that this loses the ordering, but this is not currently used.
        train_dataset = train_dataset.shuffle(seed=cfg.experiment.seed).select(range(cfg.dataset.num_examples))
    if cfg.experiment.dry_run:
        logger.info(f"Running in dry run mode -- subsampling dataset by {DRY_RUN_SUBSAMPLE_FACTOR}x")
        train_dataset = train_dataset.select(range(0, train_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR))
        eval_dataset = eval_dataset.select(range(0, eval_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR))

    # Set up custom data collator which joins examples to fill the context size
    if cfg.data_preprocessing.join_utts == "dynamic":
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer, max_seq_length=cfg.data_preprocessing.max_input_length, mlm=False)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Setting up wandb
    if not cfg.experiment.offline_run:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        if cfg.experiment.resume_checkpoint_path:
            os.environ["WANDB_RUN_ID"] = cfg.experiment.resume_run_id
            os.environ["WANDB_RESUME"] = "allow"
        wandb.init(
            entity=wandb_entity,
            project=cfg.experiment.group,
            name=cfg.experiment.name,
            config=wandb.config,
            resume="allow",
            id=cfg.experiment.resume_run_id,
        )
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
        eval_steps=cfg.trainer.max_training_steps // (2 if cfg.experiment.dry_run else 20),  # evaluate every 5% of training
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

    # Initial model evaluation
    if not cfg.experiment.resume_checkpoint_path:
        trainer.evaluate()

    # Train model
    trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

    # Evaluate best model
    trainer.stride_evaluation = 2
    trainer.evaluate(metric_key_prefix="eval_best")


if __name__ == "__main__":
    main()
