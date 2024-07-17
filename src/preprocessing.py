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

def create_phoneme_map(tokenizer, phoible_data_path=PHOIBLE_PATH):
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
        get_word_boundaries: bool = True,
    ):
        """
        Args:
            params (DataPreprocessingParams): data processing parameters
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
            get_word_boundaries (bool): whether to also output word boundaries aligned with tokens
        """

        # data processing params
        self.max_input_length = params.max_input_length
        self.join_utts = params.join_utts
        self.remove_word_boundaries = params.remove_word_boundaries

        self.tokenizer = tokenizer
        self.get_word_boundaries = get_word_boundaries
        if self.get_word_boundaries:
            self.word_boundary_token = tokenizer.convert_tokens_to_ids("WORD_BOUNDARY")
            self.utterance_boundary_token = tokenizer.convert_tokens_to_ids("UTT_BOUNDARY")

    def __call__(self, examples):

        # Make sure there's an utterance boundary at the end of each utterance
        examples["text"] = [utt + " \n" for utt in examples["text"] if utt[-1] != "\n"]

        if self.join_utts == 'static':
            batch = {}
            joined = " ".join([utt for utt in examples["text"]])
            joined = self.tokenizer(joined, truncation=False, padding=False)
            input_ids = joined["input_ids"]
            attention_mask = joined["attention_mask"]

            if self.get_word_boundaries:
                # Create an array of positions that mark the start of a word
                word_start_positions = np.minimum(len(input_ids) - 1, np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1)
                word_starts = np.zeros(len(input_ids), dtype=bool)
                word_starts[word_start_positions] = True
                # Every position after a word boundary is also a word start
                word_starts = np.logical_or(word_starts, np.array([False] + input_ids[:-1] == self.word_boundary_token))
                # Utterance boundaries are not word boundaries
                word_starts = np.logical_and(word_starts, np.array(input_ids) != self.utterance_boundary_token)
                # But the first token is always a word start
                word_starts[0] = True

            if self.remove_word_boundaries:
                mask = np.where(np.array(input_ids) != self.word_boundary_token)
                input_ids = np.array(input_ids)[mask]
                attention_mask = np.array(attention_mask)[mask]
                if self.get_word_boundaries:
                    word_starts = word_starts[mask]

            # Split the long vector into inputs of length max_input_length
            batch["input_ids"] : [input_ids[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)]
            batch["attention_mask"] : [attention_mask[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)]
            if self.get_word_boundaries:
                batch["word_starts"] = [word_starts[i : i + self.max_input_length] for i in range(0, len(input_ids), self.max_input_length)]

            return batch
        
        # If join_utts is None, we add a utterance boundary token to the start of each utterance
        elif self.join_utts is None:
            examples["text"] = [("\n " + examples["text"][i]) for i in range(len(examples["text"]))]

        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_input_length, 
            padding=False,
        )

        if self.get_word_boundaries:
            word_starts_list = []
            for i, input_ids in enumerate(tokenized["input_ids"]):
                # Create an array of positions that mark the start of a word
                line_length = len(input_ids)
                word_start_positions = np.where(np.array(input_ids) == self.word_boundary_token)[0] + 1
                word_start_positions = np.minimum(line_length - 1, word_start_positions)

                word_starts = np.zeros(len(input_ids), dtype=np.int8)
                word_starts[word_start_positions] = 1 
                word_starts = np.logical_and(word_starts, np.array(input_ids) != self.utterance_boundary_token) # Utterance boundaries are not word boundaries
                word_starts[0] = 1 # First token is always a word start
                word_starts_list.append(word_starts)

        if self.remove_word_boundaries:
            for i, input_ids in enumerate(tokenized["input_ids"]):
                mask = np.where(np.array(input_ids) != self.word_boundary_token)
                tokenized["input_ids"][i] = np.array(input_ids)[mask]
                tokenized["attention_mask"][i] = np.array(tokenized["attention_mask"][i])[mask]
                if self.get_word_boundaries:
                    word_starts_list[i] = word_starts_list[i][mask]

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        if self.get_word_boundaries:
            batch["word_starts"] = word_starts_list

        return batch
