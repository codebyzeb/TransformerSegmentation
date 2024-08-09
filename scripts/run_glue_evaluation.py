# python submodules/evaluation-pipeline-2024/finetune_classification.py --model_name_or_path $MODEL_PATH_FULL --output_dir results/finetune/$model_basename/$task/ --train_file evaluation_data/babylm_eval/glue_filtered/$TRAIN_NAME.train.jsonl --validation_file evaluation_data/babylm_eval/glue_filtered/$VALID_NAME.valid.jsonl --do_train $DO_TRAIN  --do_eval --do_predict  --use_fast_tokenizer False --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 5e-5 --num_train_epochs 10 --patience 3 --evaluation_strategy epoch --save_strategy epoch --overwrite_output_dir --trust_remote_code  --seed 12 --use_cpu True

import os
import sys
import json
import wandb

sys.path.append("submodules/evaluation-pipeline-2024")

import finetune_classification

TASKS = ["boolq", "cola", "mnli", "mnli-mm", "mrpc", "multirc", "qnli", "qqp", "rte", "sst2", "wsc"]


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_glue_evaluation.py <checkpoint_path> <resume_run_id>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    resume_run_id = sys.argv[2]

    wandb_entity = os.environ.get("WANDB_ENTITY")
    group = checkpoint_path.split("/")[1]
    name = checkpoint_path.split("/")[2]

    task_results = {}

    for task in TASKS:
        out_dir = f"checkpoints/finetune/{group}/{name}/{task}/"

        args = [
            "submodules/evaluation-pipeline-2024/finetune_classification.py",
            "--model_name_or_path",
            f"checkpoints/finetune/{group}/{name}/mnli" if task == "mnli-mm" else f"checkpoints/{group}/{name}",
            "--output_dir",
            f"{out_dir}",
            "--train_file",
            f"evaluation_data/babylm_eval/glue_filtered/{'mnli' if task == 'mnli-mm' else task}.train.jsonl",
            "--validation_file",
            f"evaluation_data/babylm_eval/glue_filtered/{task}.valid.jsonl",
            "--do_train",
            "False" if task == "mnli-mm" else "True",
            "--do_eval",
            "--do_predict",
            # "--use_fast_tokenizer",
            # "False",
            "--max_seq_length",
            "128",
            "--per_device_train_batch_size",
            "64",
            "--learning_rate",
            "5e-5",
            "--num_train_epochs",
            "10",
            "--patience",
            "3",
            "--evaluation_strategy",
            "epoch",
            # "--save_strategy",
            # "epoch",
            "--overwrite_output_dir",
            "--trust_remote_code",
            "--seed",
            "12",
            # "--use_cpu",
            # "True",
        ]

        sys.argv = args
        finetune_classification.main()

        # Read json results
        with open(out_dir + "eval_results.json", "r") as f:
            results = json.load(f)

        task_results[f"eval/glue_{task}_accuracy"] = results["eval_accuracy"]
        task_results[f"eval/glue_{task}_f1"] = results["eval_f1"]

        wandb.finish()

    print("Full task results:")
    print(task_results)

    # Log results to wandb
    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{group}/{resume_run_id}")
    config = run.config
    wandb.init(
        entity=wandb_entity,
        project=group,
        name=name,
        config=config,
        resume="allow",
        id=resume_run_id,
    )

    wandb.log(task_results)


if __name__ == "__main__":
    main()
