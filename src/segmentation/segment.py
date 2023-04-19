""" Methods for segmenting utterances using a transformer model's phoneme predictions """

import logging
import numpy as np
import pandas as pd
import scipy
import torch
import sys

sys.path.append("../")
from torch.nn import CrossEntropyLoss

from .evaluate import evaluate

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment_by_cutoff(utterance, measure, cutoff):
    """Segments an utterance by a given measure and cutoff value.
    Parameters
    ----------
    utterance : pandas.DataFrame
        A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        the prediction entropy at each point in the sequence (Entropy) and the increase in entropy at each position (Increase in Entropy).
    measure : str
        The measure to use for segmentation. One of 'Entropy', 'Increase in Entropy', 'Loss', 'Increase in Loss', 'Rank', 'Increase in Rank',
        'Boundary Prediction', 'Increase in Boundary Prediction'.
    cutoff : float
        The cutoff value to use for segmentation.

    Returns
    -------
    segmented_utterance : str
        A space-separated string of phones with ';eword' used to indicate word boundaries.
        E.g. 'w ʌ t ;eword dʒ ʌ s t ;eword h æ p ə n d ;eword d æ d i ;eword'.
    """

    segmented_utterance = " ".join(
        ";eword " + p if m > cutoff else p
        for p, m in zip(utterance.Phoneme, utterance[measure])
    ).strip()
    return segmented_utterance


def segment_by_spike(data, measure):
    """Segments an utterance by a given measure whenever there's a spike in the measure.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        the prediction entropy at each point in the sequence (Entropy) and the increase in entropy at each position (Increase in Entropy).
    measure : str
        The measure to use for segmentation. One of 'Entropy', 'Increase in Entropy', 'Loss', 'Increase in Loss', 'Rank', 'Increase in Rank',
        'Boundary Prediction', 'Increase in Boundary Prediction'.

    Returns
    -------
    segmented_utterance : str
        A space-separated string of phones with ';eword' used to indicate word boundaries.
        E.g. 'w ʌ t ;eword dʒ ʌ s t ;eword h æ p ə n d ;eword d æ d i ;eword'.
    """

    before = np.delete(np.pad(data[measure], (1, 0)), len(data))
    after = np.delete(np.pad(data[measure], (0, 1)), 0)
    boundaries = np.logical_and(data[measure] > before, data[measure] > after)
    segmented_utterance = " ".join(
        ";eword " + p if b else p for p, b in zip(data.Phoneme, boundaries)
    ).strip()
    return segmented_utterance


class Segmenter(object):
    def __init__(self, model, tokenizer, utterances):
        """A class for segmenting utterances using a transformer model's phoneme predictions.
        Parameters
        ----------
        model : torch.nn.Module
            A transformer model.
        tokenizer : AutoTokenizer
            A tokenizer for the model.
        utterances : list
            A list of utterances to segment.
        """

        self.max_sequence_length = 128
        self.model = model
        self.tokenizer = tokenizer
        self.boundary_token = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("UTT_BOUNDARY")
        )[0]

        self.gold_utterances = [
            line.strip()
            for line in utterances
            if len([phone for phone in line.strip(" ") if phone != ";eword"])
            <= self.max_sequence_length - 1
        ]

        self.processed_utterances = []
        for utt in self.gold_utterances:
            self.processed_utterances.append(self.process_utterance(utt))

    def process_utterance(self, utterance):
        """Processes an utterance into a dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and a variety of measures of prediction uncertainty at each point in the sequence (Entropy, Increase in Entropy, Loss, Increase in Loss, Rank, Increase in Rank).
        Parameters
        ----------
        utterance : str
            A space-separated string of phones with ';eword' used to indicate word boundaries.
            E.g. 'w ʌ t ;eword dʒ ʌ s t ;eword h æ p ə n d ;eword d æ d i ;eword'.

        Returns
        -------
        data : pandas.DataFrame
            A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
            and a variety of measures of prediction uncertainty at each point in the sequence (Entropy, Increase in Entropy, Loss, Increase in Loss, Rank, Increase in Rank).
        """

        processed = utterance.split(" ")
        phonemes = []
        word_starts = []
        next_word = True
        for c in processed:
            if c == ";eword":
                next_word = True
            else:
                phonemes.append(c)
                word_starts.append(next_word)
                next_word = False

        data = self.get_uncertainties(
            ["UTT_BOUNDARY"] + phonemes
        )  # Might want to add <START> token if using TruncateTokenizer
        data["Pos"] = list(range(len(word_starts)))
        data["Starts"] = word_starts
        data["Phoneme"] = phonemes
        return pd.DataFrame(data)

    def get_uncertainties(self, utterance):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence.
        Parameters
        ----------
        utterance : list of str
            A list of phones.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        # Token id tensor
        token_ids = self.tokenizer.convert_tokens_to_ids(utterance)
        uncertainties = {
            "Entropy": [],
            "Increase in Entropy": [],
            "Loss": [],
            "Increase in Loss": [],
            "Rank": [],
            "Increase in Rank": [],
            "Boundary Prediction": [],
            "Increase in Boundary Prediction": [],
        }
        entropy = 0
        boundary_prediction = 0
        loss = 0
        rank = 0

        with torch.no_grad():
            input = torch.tensor([token_ids], dtype=torch.long)
            logits = self.model(input, labels=input).logits.detach()[0][:-1]
            loss_fct = CrossEntropyLoss(reduction="none")

            loss = loss_fct(logits, input[0][1:]).detach().numpy()
            increase_in_loss = np.insert(loss[1:] - loss[:-1], 0, 0)
            entropy = (
                torch.distributions.Categorical(logits=logits)
                .entropy()
                .detach()
                .numpy()
            )
            increase_in_entropy = np.insert(entropy[1:] - entropy[:-1], 0, 0)
            rank = np.log2(
                1
                + (
                    logits.argsort(descending=True)
                    == input[0][1:].unsqueeze(1)
                ).nonzero(as_tuple=True)[1]
            )
            increase_in_rank = np.insert(rank[1:] - rank[:-1], 0, 0)
            boundary_prediction = (
                torch.softmax(logits, dim=1)[:, self.boundary_token]
                .detach()
                .numpy()
            )
            increase_in_boundary_prediction = np.insert(
                boundary_prediction[1:] - boundary_prediction[:-1], 0, 0
            )

            uncertainties["Entropy"] = entropy
            uncertainties["Increase in Entropy"] = increase_in_entropy
            uncertainties["Loss"] = loss
            uncertainties["Increase in Loss"] = increase_in_loss
            uncertainties["Rank"] = rank
            uncertainties["Increase in Rank"] = increase_in_rank
            uncertainties["Boundary Prediction"] = boundary_prediction
            uncertainties[
                "Increase in Boundary Prediction"
            ] = increase_in_boundary_prediction

        return uncertainties

    def evaluate_cutoff_segmentation(self, measure, cutoffs):
        """Given a measure and a range of cutoffs, segments according to the cutoff and evaluates the results.
        Parameters
        ----------
        measure : str
            The measure to use for segmentation.
        cutoffs : list
            A list of cutoffs to use for segmentation.

        Returns
        -------
        results : pandas.DataFrame
            A dataframe containing the results of segmentation for each cutoff.
        """

        all_results = []
        for cutoff in cutoffs:
            segmented_utterances = [
                segment_by_cutoff(utt, measure, cutoff)
                for utt in self.processed_utterances
            ]
            results = evaluate(segmented_utterances, self.gold_utterances)
            results["Cutoff"] = cutoff
            all_results.append(results)

        logging.info(
            "Segmented utterances for a range of cutoffs using measure: {}".format(
                measure
            )
        )
        return pd.DataFrame(all_results)

    def evaluate_spike_segmentation(self, measure):
        """Given a measure, segments according to spikes in the measure and evaluates the results.
        Parameters
        ----------
        measure : str
            The measure to use for segmentation.

        Returns
        -------
        results : pandas.DataFrame
            A dataframe containing the results of segmentation.
        """

        segmented_utterances = [
            segment_by_spike(utt, measure) for utt in self.processed_utterances
        ]

        logging.info(
            "Segmented utterances according to spikes in measure: {}".format(
                measure
            )
        )
        return evaluate(segmented_utterances, self.gold_utterances)
