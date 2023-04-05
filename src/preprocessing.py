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
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length",
        )
        return tokenized
