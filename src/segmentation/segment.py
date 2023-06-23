""" Methods for segmenting utterances using a transformer model's phoneme predictions """

import logging

import numpy as np
import pandas as pd
import torch
from evaluate import load
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

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
        A space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
        E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.
    """

    segmented_utterance = " ".join("WORD_BOUNDARY " + p if m > cutoff else p for p, m in zip(utterance.Phoneme, utterance[measure])).strip()
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
        A space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
        E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.
    """

    before = np.delete(np.pad(data[measure], (1, 0)), len(data))
    after = np.delete(np.pad(data[measure], (0, 1)), 0)
    boundaries = np.logical_and(data[measure] > before, data[measure] > after)
    segmented_utterance = " ".join("WORD_BOUNDARY " + p if b else p for p, b in zip(data.Phoneme, boundaries)).strip()
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
        self.boundary_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("UTT_BOUNDARY"))[0]

        self.gold_utterances = [" ".join([p for p in utt.strip().split(" ") if p != ""]) for utt in utterances]

        self.processed_utterances = []
        for utt in self.gold_utterances:
            self.processed_utterances.append(self.process_utterance(utt))

        self.measures = self.processed_utterances[0].columns[:-3].tolist()

        self.metric = load("transformersegmentation/segmentation_scores")

    def process_utterance(self, utterance):
        """Processes an utterance into a dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and a variety of measures of prediction uncertainty at each point in the sequence (Entropy, Increase in Entropy, Loss, Increase in Loss, Rank, Increase in Rank).
        Parameters
        ----------
        utterance : str
            A space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
            E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.

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
            if c == " ":
                continue
            if c == "WORD_BOUNDARY":
                next_word = True
            else:
                phonemes.append(c)
                word_starts.append(next_word)
                next_word = False

        data = self.get_uncertainties(["UTT_BOUNDARY"] + phonemes)
        data["Pos"] = list(range(len(word_starts)))
        data["Starts"] = word_starts
        data["Phoneme"] = phonemes
        return pd.DataFrame(data)

    def get_uncertainties(self, utterance):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence,
            must be implemented by subclass.

        Parameters
        ----------
        utterance : list of str
            A list of phones.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        raise NotImplementedError

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
            segmented_utterances = [segment_by_cutoff(utt, measure, cutoff) for utt in self.processed_utterances]
            results = self.metric.compute(
                predictions=segmented_utterances,
                references=self.gold_utterances,
            )
            results["Cutoff"] = cutoff
            all_results.append(results)

        logging.info("Segmented utterances for a range of cutoffs using measure: {}".format(measure))
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

        segmented_utterances = [segment_by_spike(utt, measure) for utt in self.processed_utterances]

        logging.info("Segmented utterances according to spikes in measure: {}".format(measure))
        return self.metric.compute(predictions=segmented_utterances, references=self.gold_utterances)


class GPT2Segmenter(Segmenter):
    def get_uncertainties(self, utterance):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence,
            extracted from a GPT2 model.

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

        with torch.no_grad():
            input = torch.tensor([token_ids], dtype=torch.long).to(DEFAULT_DEVICE)
            logits = self.model(input, labels=input).logits.detach()[0][:-1]
            loss_fct = CrossEntropyLoss(reduction="none")

            loss = loss_fct(logits, input[0][1:]).detach().cpu().numpy()
            increase_in_loss = np.insert(loss[1:] - loss[:-1], 0, 0)
            entropy = torch.distributions.Categorical(logits=logits).entropy().detach().cpu().numpy()
            increase_in_entropy = np.insert(entropy[1:] - entropy[:-1], 0, 0)
            rank = np.log2(1 + (logits.argsort(descending=True) == input[0][1:].unsqueeze(1)).nonzero(as_tuple=True)[1].detach().cpu())
            increase_in_rank = np.insert(rank[1:] - rank[:-1], 0, 0)
            boundary_prediction = torch.softmax(logits, dim=1)[:, self.boundary_token].detach().cpu().numpy()
            increase_in_boundary_prediction = np.insert(boundary_prediction[1:] - boundary_prediction[:-1], 0, 0)

            uncertainties = {}
            uncertainties["Entropy"] = entropy
            uncertainties["Increase in Entropy"] = increase_in_entropy
            uncertainties["Loss"] = loss
            uncertainties["Increase in Loss"] = increase_in_loss
            uncertainties["Rank"] = rank
            uncertainties["Increase in Rank"] = increase_in_rank
            uncertainties["Boundary Prediction"] = boundary_prediction
            uncertainties["Increase in Boundary Prediction"] = increase_in_boundary_prediction

        return uncertainties


class GPT2FeaturesSegmenter(GPT2Segmenter):
    def get_uncertainties(self, utterance):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence,
            extracted from a GPT2 features model.

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
        return_token_logits_tmp = self.model.return_token_logits
        self.model.return_token_logits = True
        uncertainties = super().get_uncertainties(utterance)
        self.model.return_token_logits = False

        with torch.no_grad():
            input = torch.tensor([token_ids], dtype=torch.long).to(DEFAULT_DEVICE)
            logits = self.model(input, labels=input).logits.detach()[0, :-1,...].permute(1, 2, 0)
            loss_fct = CrossEntropyLoss(reduction="none")

            # For this model, we get a loss per feature, per position
            full_loss = torch.zeros_like(logits[..., 0])
            label_vectors = self.model.feature_map.as_indices(input[0])[1:].long().T
            full_loss = loss_fct(logits, label_vectors) / self.model.feature_size
            full_loss = full_loss.detach().cpu().numpy()
            loss = full_loss.mean(axis=0)
            increase_in_loss = np.insert(loss[1:] - loss[:-1], 0, 0)

            # We get the average entropy for each feature across positions
            feature_entropy = np.array(
                [torch.distributions.Categorical(logits=logits[..., i]).entropy().mean().detach().cpu() for i in range(logits.shape[2])]
            )
            increase_in_entropy = np.insert(feature_entropy[1:] - feature_entropy[:-1], 0, 0)

            # Standard deviation of the loss
            loss_std = full_loss.std(axis=0)
            increase_in_loss_std = np.insert(loss_std[1:] - loss_std[:-1], 0, 0)

            # Boundary is predicted by the last feature being equal to 1
            boundary_prediction = torch.softmax(logits, dim=1)[-1, 1].detach().cpu().numpy()
            increase_in_boundary_prediction = np.insert(boundary_prediction[1:] - boundary_prediction[:-1], 0, 0)

            uncertainties["Feature Loss"] = loss
            uncertainties["Increase in Feature Loss"] = increase_in_loss
            uncertainties["Feature Entropy"] = feature_entropy
            uncertainties["Increase in Feature Entropy"] = increase_in_entropy
            uncertainties["Loss Deviation"] = loss_std
            uncertainties["Increase in Loss Deviation"] = increase_in_loss_std
            uncertainties["Boundary Feature Prediction"] = boundary_prediction
            uncertainties["Increase in Boundary Feature Prediction"] = increase_in_boundary_prediction

        self.model.return_token_logits = return_token_logits_tmp

        return uncertainties
