"""Class for preprocessing the data, including tokenization, etc."""

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer

from .config import DataPreprocessingParams

FEATURES = ['tone', 'stress', 'syllabic', 'short', 'long',
       'consonantal', 'sonorant', 'continuant', 'delayedRelease',
       'approximant', 'tap', 'trill', 'nasal', 'lateral', 'labial', 'round',
       'labiodental', 'coronal', 'anterior', 'distributed', 'strident',
       'dorsal', 'high', 'low', 'front', 'back', 'tense',
       'retractedTongueRoot', 'advancedTongueRoot', 'periodicGlottalSource',
       'epilaryngealSource', 'spreadGlottis', 'constrictedGlottis', 'fortis',
       'lenis', 'raisedLarynxEjective', 'loweredLarynxImplosive', 'click']

PAD_FEATURE_VEC = [0] * len(FEATURES) + [0]
BOUNDARY_FEATURE_VEC = [0] * len(FEATURES) + [1]
UNK_FEATURE_VEC = [0] * len(FEATURES) + [2]

PHOIBLE_PATH = 'data/phoible.csv'

def create_phoneme_map(tokenizer, language, phoible_data_path=PHOIBLE_PATH):
        """
        Creates a map from tokenizer IDs to features.
        """
        
        phoible = pd.read_csv(phoible_data_path)
        phoneme_map = {}

        for phoneme, id in tokenizer.vocab.items():
            row = phoible[phoible['Phoneme'] == phoneme][FEATURES]
            
            # Convert features to a vector of 0s, 1s, and 2s
            if row.shape[0] != 0:
                features = [1 if f == '-' else 2 if f == '+' else 0 for f in row.values[0]] + [0]
            elif phoneme in ['WORD_BOUNDARY', 'UTT_BOUNDARY']:
                features = BOUNDARY_FEATURE_VEC
            elif phoneme in ['PAD', 'EOS', 'BOS']:
                features = PAD_FEATURE_VEC
            else:
                features = UNK_FEATURE_VEC
            phoneme_map[id] = features
        return phoneme_map

class DataPreprocessor(object):
    def __init__(
        self,
        params: DataPreprocessingParams,
        tokenizer: PreTrainedTokenizer,
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
            batch = {"input_ids": [], "attention_mask": [], "word_starts": []}
            for i in range(0, len(input_ids), self.max_input_length):
                batch["input_ids"].append(input_ids[i : i + self.max_input_length])
                batch["attention_mask"].append(attention_mask[i : i + self.max_input_length])
                batch["word_starts"].append(word_starts[i : i + self.max_input_length])
            return batch

        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_input_length * 2,  # Since we're removing tokens, we need to insert additional pads
            padding="max_length",
        )

        word_starts_list = []

        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Create an array of positions that mark the start of a word
            line_length = len(input_ids)
            word_start_positions = np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1
            word_start_positions = np.minimum(line_length - 1, word_start_positions)
            word_starts = np.zeros(len(input_ids), dtype=np.int8)
            word_starts[word_start_positions] = 1

            # Remove the word boundary tokens and truncate
            mask = np.where(np.array(input_ids) != self.word_boundary_token)
            tokenized["input_ids"][i] = np.array(input_ids)[mask][:self.max_input_length]
            tokenized["attention_mask"][i] = np.array(tokenized["attention_mask"][i])[mask][:self.max_input_length]
            word_starts_list.append(word_starts[mask][:self.max_input_length])

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "word_starts": word_starts_list,
        }
        return batch
