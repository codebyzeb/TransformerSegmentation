""" 
This script will run the BabySLM evaluation pipeline to return a lexical and syntactic score for the model. Instead of running their
load_stimuli_text function, we have already saved the stimuli in a csv file in data/babyslm. This script loads the stimuli and 
runs the model to get a score for each stimuli. It then calls the BabySLM evaluation function to get the final score.

"""

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import tqdm

sys.path.append("submodules")

from BabySLM.scripts.metrics import compute_lexical, compute_syntactic

LEXICAL_STIMULI = "data/babyslm/lexical/dev/lexical_stimuli.csv"
SYNTACTIC_STIMULI = "data/babyslm/syntactic/dev/syntactic_stimuli.csv"
LEXICAL_GOLD_DATA = "data/babyslm/lexical/dev/gold.csv"
SYNTACTIC_GOLD_DATA = "data/babyslm/syntactic/dev/gold.csv"


def extract_probabilities(examples, model, tokenizer):
    tokenized = [tokenizer(transcription, return_tensors="pt") for transcription in examples]
    probabilities = []
    for i in tqdm.tqdm(range(len(examples))):
        input_ids = tokenized[i]["input_ids"].to(model.device)
        attention_mask = tokenized[i]["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
            probabilities.append(-outputs.loss.item())
    return probabilities


def write_probabilities(seq_names, probabilities, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        for filename, prob in zip(seq_names, probabilities):
            f.write(f"{filename} {prob}\n")


def babyslm_evaluation(model, tokenizer, model_path, type):
    """
    Returns either the lexical or syntactic score for the model.

    Args:
        model: The model to be evaluated
        tokenizer: The tokenizer for the model
        model_path: The path to the model checkpoint
        type: The type of evaluation to be done. Either 'lexical' or 'syntactic'
    """

    if type == "lexical":
        stimuli = pd.read_csv(LEXICAL_STIMULI)
    elif type == "syntactic":
        stimuli = pd.read_csv(SYNTACTIC_STIMULI)
    else:
        raise ValueError("type must be either lexical or syntactic")

    logging.info(f"Running BabySLM evaluation for {type} stimuli")

    # Some slight adjustments needed for the stimuli
    stimuli["transcription"] = stimuli["transcription"].str.replace("tʃ", "t̠ʃ")
    stimuli["transcription"] = stimuli["transcription"].str.replace("dʒ", "d̠ʒ")

    # Add new lines to the beginning and end of the transcription
    stimuli["transcription"] = "\n " + stimuli["transcription"] + "\n"

    # Get probabilities for each example and write to a file
    examples = stimuli["transcription"].tolist()
    probabilities = extract_probabilities(examples, model, tokenizer)
    seq_names = stimuli["filename"].tolist()
    out_file = model_path / "babyslm" / f"{type}.txt"
    write_probabilities(seq_names, probabilities, out_file)

    # Run evaluation script on computed probabilities
    if type == "lexical":
        gold_file = Path(LEXICAL_GOLD_DATA)
        _, by_pair, _, _ = compute_lexical.evaluate(gold_file, out_file, is_text=True)
        accuracy = by_pair["score"].mean()
        logging.info(f"Lexical accuracy: {accuracy}")
    else:
        gold_file = Path(SYNTACTIC_GOLD_DATA)
        _, by_pair, _ = compute_syntactic.evaluate(gold_file, out_file, is_text=True)
        accuracy = by_pair["score"].mean()
        logging.info(f"Syntactic accuracy: {accuracy}")

    return accuracy
