"""Class for preprocessing the data, including tokenization, etc."""

import numpy as np

from transformers import PreTrainedTokenizer

from .config import DataPreprocessingParams


class DataPreprocessor(object):
    def __init__(
        self,
        params: DataPreprocessingParams,
        tokenizer: PreTrainedTokenizer,
        labels_are_word_boundaries: bool = False,
    ):
        """
        Args:
            cfg (TransformerSegmentationConfig): hydra config object
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
        """

        # data processing params
        self.max_input_length = params.max_input_length
        self.join_utts = params.join_utts
        self.tokenizer = tokenizer
        self.labels_are_word_boundaries = labels_are_word_boundaries
        self.word_boundary_token = tokenizer.convert_tokens_to_ids("WORD_BOUNDARY")

    def __call__(self, examples):
        if self.join_utts:
            joined = " ".join([utt for utt in examples["text"]])
            joined = self.tokenizer(joined, truncation=False, padding=False)
            input_ids = joined["input_ids"]
            attention_mask = joined["attention_mask"]

            # Create an array of positions that mark the start of a word
            word_start_positions = np.minimum(len(input_ids) - 1, np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1)
            word_starts = np.zeros(len(input_ids), dtype=bool)
            word_starts[word_start_positions] = True

            # Remove the word boundary tokens
            mask = np.where(np.array(input_ids) != self.word_boundary_token)
            input_ids = np.array(input_ids)[mask]
            attention_mask = np.array(attention_mask)[mask]
            word_starts = word_starts[mask]

            # Split the long vector into inputs of length max_input_length
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for i in range(0, len(input_ids), self.max_input_length):
                batch["input_ids"].append(
                    input_ids[i : i + self.max_input_length]
                )
                batch["attention_mask"].append(
                    attention_mask[i : i + self.max_input_length]
                )
                if self.labels_are_word_boundaries:
                    batch["labels"].append(
                        word_starts[i : i + self.max_input_length]
                    )
                else:
                    batch["labels"].append(
                        input_ids[i : i + self.max_input_length]
                    )
            return batch
        
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length",
        )

        tokenized['labels'] = tokenized['input_ids'].copy()

        for i in range(len(tokenized["input_ids"])):

            # Create an array of positions that mark the start of a word
            word_start_positions = np.minimum(len(tokenized["input_ids"][i]) - 1, np.where(np.array(tokenized["input_ids"][i]) == self.word_boundary_token)[0] + 1)
            word_starts = np.zeros(len(tokenized["input_ids"][i]), dtype=bool)
            word_starts[word_start_positions] = True

            # Remove the word boundary tokens
            mask = np.where(np.array(tokenized["input_ids"][i]) != self.word_boundary_token)
            tokenized["input_ids"][i] = np.array(tokenized["input_ids"][i])[mask]
            tokenized["attention_mask"][i] = np.array(tokenized["attention_mask"][i])[mask]
            if self.labels_are_word_boundaries:
                tokenized["labels"][i] = word_starts[mask]
            else:
                tokenized["labels"][i] = tokenized["input_ids"][i]

            tokenized["input_ids"][i] = tokenized["input_ids"][i]
            tokenized["attention_mask"][i] = tokenized["attention_mask"][i]
            tokenized["labels"][i] = tokenized["labels"][i]

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
        }
        return batch
