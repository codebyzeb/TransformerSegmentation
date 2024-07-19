""" 
This script will run the BabyLM evaluation pipeline to return scores for each BLIMP task. The results are also saved
to a json file in the model path. Ensure that the blimp tasks are provided in evaluation_data/blimp_filtered. 
"""

import logging
import json

import numpy as np

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

def blimp_evaluation(model, tokenizer, model_path, batch_size, tasks, device, is_phonemized=False):
    """ Run BLIMP evaluation using BabyLM evaluation pipline """

    model_wrapper = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    task_list = tasks.split(",")
    if is_phonemized:
        task_manager = lm_eval.tasks.TaskManager(override_data_path="evaluation_data/blimp_filtered_phonemized")
    else:
        task_manager = lm_eval.tasks.TaskManager()
    task_names = task_manager.match_tasks(task_list)

    logging.info(f"Running BLIMP evaluation for {task_names}")
    results = lm_eval.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=0,
        task_manager=task_manager,
        batch_size='auto',
        device=device,
    )

    if results is not None:
        logging.info(make_table(results))

    if "groups" in results:
        logging.info(make_table(results, "groups"))

    # Write json to model path
    with open(model_path / "blimp_results.json", "w") as f:
        json.dump(results, indent=2, default=_handle_non_serializable, ensure_ascii=False, fp=f)

    return results

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)