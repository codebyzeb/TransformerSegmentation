"""Class for preprocessing the data, including tokenization, etc."""

import string

from transformers import PreTrainedTokenizer

from .config import TransformerSegmentationConfig


class DataPreprocessor(object):
    def __init__(
        self,
        cfg: TransformerSegmentationConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Args:
            cfg (TransformerSegmentationConfig): hydra config object
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
        """

        # data processing params
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.join_utts = cfg.data_preprocessing.join_utts
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # for callback_function in self.callback_functions:
        #     examples[callback_function] = getattr(self, callback_function)(
        #         examples["text"]
        #     )

        # tokenize the input text
        # return self.tokenizer(
        #     examples["text"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_input_length,
        #     return_special_tokens_mask=False,
        # )
        if self.join_utts:
            joined = " ".join([utt for utt in examples["text"]])
            joined = self.tokenizer(joined, truncation=False, padding=False)
            input_ids = joined["input_ids"]
            attention_mask = joined["attention_mask"]
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for i in range(0, len(input_ids), self.max_input_length):
                batch["input_ids"].append(
                    input_ids[i : i + self.max_input_length]
                )
                batch["attention_mask"].append(
                    attention_mask[i : i + self.max_input_length]
                )
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
        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"],
        }
        return batch
