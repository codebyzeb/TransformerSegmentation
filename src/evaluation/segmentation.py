""" Methods for segmenting utterances using a transformer model's phoneme predictions """

import logging

import numpy as np
import pandas as pd
import torch
from evaluate import load
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import minimize_scalar

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment_by_cutoff(data, measure, cutoff):
    """Segments a sequence of phonemes by a given measure and cutoff value.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and values for a variety of measures of prediction uncertainty at each point in the sequence.
    measure : str
        The measure to use for segmentation. Must be a column in `data`.
    cutoff : float
        The cutoff value to use for segmentation.

    Returns
    -------
    segmented : str
        A space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
        E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.
    """

    segmented = " ".join("WORD_BOUNDARY " + p if m > cutoff else p for p, m in zip(data.Phoneme, data[measure])).strip()
    return segmented

def segment_by_spike(data, measure):
    """Segments a sequence of phonemes by a given measure whenever there's a spike in the measure.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and values for a variety of measures of prediction uncertainty at each point in the sequence.
    measure : str
        The measure to use for segmentation. Must be a column in `data`.

    Returns
    -------
    segmented : str
        A space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
        E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.
    """

    before = np.delete(np.pad(data[measure], (1, 0)), len(data))
    after = np.delete(np.pad(data[measure], (0, 1)), 0)
    boundaries = np.logical_and(data[measure] > before, data[measure] > after)
    segmented = " ".join("WORD_BOUNDARY " + p if b else p for p, b in zip(data.Phoneme, boundaries)).strip()
    return segmented

def get_gold_segmentation(data):
    """ Extracts the gold segmentation from the data.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and values for a variety of measures of prediction uncertainty at each point in the sequence.

    Returns
    -------
    segmented_utterances : list of str
        A list of space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
        E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.
    """

    segmented = " ".join("WORD_BOUNDARY " + p if s else p for p, s in zip(data.Phoneme, data.Starts)).strip()
    return segmented.split(" UTT_BOUNDARY ")

class Segmenter(object):
    def __init__(self, model, tokenizer, utterances, max_sequence_length=64, batch_size=16, stride=10):
        """A class for segmenting utterances using a transformer model's phoneme predictions.
        Parameters
        ----------
        model : torch.nn.Module
            A transformer model.
        tokenizer : AutoTokenizer
            A tokenizer for the model.
        utterances : list
            A list of utterances to segment.
        max_sequence_length : int
            The maximum sequence length to use when extracting predictions from the model.
        batch_size : int
            The batch size to use when extracting predictions from the model.
        stride : int
            The stride to use when extracting predictions from the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.stride = stride

        self.boundary_token = tokenizer.eos_token_id

        utterances = [" ".join([p for p in utt.strip().split(" ") if p != ""]) for utt in utterances] # Remove empty phones

        # Process utterances and get gold segmentation
        self.processed_utterances = self.process_utterances(utterances)
        self.gold_utterances = get_gold_segmentation(self.processed_utterances)

        self.measures = self.processed_utterances.columns[:-3].tolist()

        self.metric = load("transformersegmentation/segmentation_scores")

    def process_utterances(self, utterances):
        """Processes each utterance in `utterances` into a dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
        and a variety of measures of prediction uncertainty at each point in the sequence (Entropy, Increase in Entropy, Loss, Increase in Loss, Rank, Increase in Rank).
        Parameters
        ----------
        utterances : list of str
            A list of space-separated string of phones with 'WORD_BOUNDARY' used to indicate word boundaries.
            E.g. 'w ʌ t WORD_BOUNDARY dʒ ʌ s t WORD_BOUNDARY h æ p ə n d WORD_BOUNDARY d æ d i WORD_BOUNDARY'.

        Returns
        -------
        data : pandas.DataFrame
            A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
            and a variety of measures of prediction uncertainty at each point in the sequence, depending on the model.
        """

        phonemes = []
        word_starts = []
        next_word = True
        for utterance in utterances:
            processed = utterance.split(" ")
            for c in processed:
                if c == " ":
                    continue
                if c == "WORD_BOUNDARY":
                    next_word = True
                else:
                    phonemes.append(c)
                    word_starts.append(next_word)
                    next_word = False
            phonemes.append("UTT_BOUNDARY")
            word_starts.append(False)

        data = self.get_uncertainties(phonemes)
        data["Pos"] = list(range(len(word_starts)))
        data["Starts"] = word_starts
        data["Phoneme"] = phonemes
        return pd.DataFrame(data)

    def get_uncertainties(self, phonemes):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence,
            must be implemented by subclass.

        Parameters
        ----------
        phonemes : list of str
            A list of phones.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        raise NotImplementedError

    def evaluate_cutoff_segmentation(self, measure, cutoff, subsample=1):
        """Given a measure and a cutoff, segments by adding word boundaries whenever the measure is above the cutoff and evaluates the results.
        
        Parameters
        ----------
        measure : str
            The measure to use for segmentation.
        cutoffs : float
            The cutoff value used to place word boundaries in the segmentation.

        Returns
        -------
        results : pandas.DataFrame
            A dataframe containing the results of segmentation.
        """

        segmented_utterances = segment_by_cutoff(self.processed_utterances, measure, cutoff).split(" UTT_BOUNDARY ")
        results = self.metric.compute(
            predictions=segmented_utterances[::subsample],
            references=self.gold_utterances[::subsample],
        )

        logging.debug("Segmented utterances using a cutoff of {} for measure: {}".format(cutoff, measure))
        return results
    
    def find_best_cutoff(self, measure, score):
        """Given a measure and a score, finds the cutoff value that maximises the segmentation score.

        Subsamples processed_utterances by 10 to speed up the search.

        Parameters
        ----------
        measure : str
            The measure to use for segmentation.
        score : str
            The score to use for evaluation.

        Returns
        -------
        best_cutoff : float
            The best cutoff value for segmentation.
        best_score : float
            The resulting score for segmentation.
        """
        
        min, max = self.processed_utterances[measure].min(), self.processed_utterances[measure].max()
        fun = lambda x: -self.evaluate_cutoff_segmentation(measure=measure, cutoff=x, subsample=10)[score]
        result = minimize_scalar(fun, bounds=(min, max), method='bounded')
        logging.debug("Found best cutoff for measure {} using score {}: {}".format(measure, score, result.x))
        final_score = self.evaluate_cutoff_segmentation(measure=measure, cutoff=result.x)[score]
        return result.x, final_score

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

        segmented_utterances = segment_by_spike(self.processed_utterances, measure).split(" UTT_BOUNDARY ")

        logging.debug("Segmented utterances according to spikes in measure: {}".format(measure))
        return self.metric.compute(predictions=segmented_utterances, references=self.gold_utterances)
    
    def add_majority_vote(self, measure_cutoffs):
        """ Given a set of measures and their respective cutoff values, calculates two types of majority votes
        and adds them to the processed_utterance data.
        
        * The Majority Vote Cutoff measure indicates how many measures exceed their respective cutoffs at each position.
        * The Majority Vote Spike measure indicates how many measures spike at each position.
        
        These measures are then normalised so that a vote above 0.5 indicates that the majority of measures agree.

        Parameters
        ----------
        measures_cutoffs : dict
            A dictionary containing the measure names as keys and their respective cutoff values as values.
        """

        self.processed_utterances['Majority Vote Cutoff'] = 0
        self.processed_utterances['Majority Vote Spike'] = 0
        for measure, cutoff in measure_cutoffs.items():
            self.processed_utterances['Majority Vote Cutoff'] += (self.processed_utterances[measure] > cutoff).astype(int)
            shift_left = self.processed_utterances[measure].shift(1)
            shift_right = self.processed_utterances[measure].shift(-1)
            self.processed_utterances['Majority Vote Spike'] += ((self.processed_utterances[measure] > shift_left) & (self.processed_utterances[measure] > shift_right)).astype(int)
        num_measures = len(measure_cutoffs)
        self.processed_utterances['Majority Vote Cutoff'] /= num_measures
        self.processed_utterances['Majority Vote Spike'] /= num_measures

class GPT2Segmenter(Segmenter):
    def get_uncertainties(self, phonemes):
        """ Gets a variety of measures of prediction uncertainty at each point in the sequence,
            extracted from a GPT2 model.

        Parameters
        ----------
        phonemes : list of str
            A list of phonemes.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        # Prepare token ids tensor and get logits. We add a boundary token at the beginning 
        # to get the prediction for the first phoneme and we discard the last prediction.
        token_ids = [self.boundary_token] + self.tokenizer.convert_tokens_to_ids(phonemes)
        long_input_ids = torch.tensor(token_ids, dtype=torch.long).to(DEFAULT_DEVICE)
        logits = self.get_logits(long_input_ids)[:len(token_ids) - 1]

        logging.info(f'Computing uncertainty measures...')
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, long_input_ids[1:]).detach().cpu().numpy()
        increase_in_loss = np.insert(loss[1:] - loss[:-1], 0, 0)
        entropy = torch.distributions.Categorical(logits=logits).entropy().detach().cpu().numpy()
        increase_in_entropy = np.insert(entropy[1:] - entropy[:-1], 0, 0)
        rank = np.log2(1 + (logits.argsort(descending=True) == long_input_ids[1:].unsqueeze(1)).nonzero(as_tuple=True)[1].detach().cpu())
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

    def get_logits(self, long_input_ids):
        """ Gets logits by running the model on the long vector of phonemes.
        Parameters
        ----------
        long_input_ids : torch.Tensor
            A tensor of token ids.
        
        Returns
        -------
        logits : torch.Tensor
            A tensor of logits.
        """

        if self.stride is None or self.stride == 0:
            return self.get_logits_individual(long_input_ids)
        else:
            return self.get_logits_stride(long_input_ids)
    
    def get_logits_individual(self, long_input_ids):
        """ Gets logits by running the model on each utterance individually, which is faster but provides less context for early predictions. 
        Parameters
        ----------
        long_input_ids : torch.Tensor
            A tensor of token ids.

        Returns
        -------
        logits : torch.Tensor
            A tensor of logits.
        """

        logging.info(f'Extracting logits for all {len(long_input_ids)} phonemes in segmentation set by feeding in each utterance to the model individually...')

        input_ids = []
        input_id_length = len(long_input_ids)
        prev = 0
        for begin_loc in range(1, len(long_input_ids)):
            if long_input_ids[begin_loc] == self.boundary_token:
                inputs = long_input_ids[prev:begin_loc+1].to(DEFAULT_DEVICE) # Include both utterance boundaries for every input. We remove the last prediction later.
                input_ids.append(inputs)
                prev = begin_loc
        if prev < input_id_length:
            inputs = long_input_ids[prev:].to(DEFAULT_DEVICE)
            input_ids.append(inputs)

        # Feed in each utterance individually (could pad and batch to make quicker)
        logits = []
        logging.info('Extracting logits...')
        for inputs in tqdm(input_ids):
            with torch.no_grad():
                outputs = self.model(inputs.unsqueeze(0)).logits.detach()
                logits.append(outputs[0][:-1]) # Remove the last prediction here. This ensures we have one prediction per phoneme in the long_input_ids tensor.

        return torch.cat(logits)

    def get_logits_stride(self, long_input_ids):
        """ Gets logits by running the model on the long vector of phonemes, shifting by a stride to provide maximum context.
        Parameters
        ----------
        long_input_ids : torch.Tensor
            A tensor of token ids.

        Returns
        -------
        logits : torch.Tensor
            A tensor of logits.
        """

        logging.info(f'Extracting logits for all {len(long_input_ids)} phonemes in segmentation set using stride {self.stride} and max length {self.max_sequence_length}...')

        # Batch the input ids and labels into sequences of length max_length by shifting by stride
        input_ids = []
        input_id_length = len(long_input_ids)
        for begin_loc in range(0, len(long_input_ids), self.stride):
            end_loc = min(begin_loc + self.max_sequence_length, len(long_input_ids))
            inputs = long_input_ids[begin_loc:end_loc].to(DEFAULT_DEVICE)
            input_ids.append(inputs)
            if end_loc == input_id_length:
                break

        # Pad final stride with 0s
        input_ids[-1] = torch.cat((input_ids[-1], torch.zeros(self.max_sequence_length - len(input_ids[-1]), dtype=torch.long).to(DEFAULT_DEVICE)))

        # Ensure divisible by batch_size
        while len(input_ids) % self.batch_size != 0:
            input_ids.append(torch.zeros_like(input_ids[0]))

         # Stack into batches of batch_size
        input_ids = torch.stack(input_ids)
        input_ids = input_ids.view(-1, self.batch_size, self.max_sequence_length)
        seq_len = input_ids.size(0)

        # Get logits from each batch and concatenate them
        logits = []
        first = True
        logging.info('Extracting logits...')
        for i in tqdm(range(seq_len)):
            with torch.no_grad():
                outputs = self.model(input_ids[i]).logits.detach()
                for batch in range(self.batch_size):
                    if first:
                        logits = outputs[batch]
                        first = False
                    else:
                        # After the first batch, we only want the last stride tokens (which are newly predicted)
                        logits = torch.cat((logits, outputs[batch][-self.stride:]))

        return logits

class GPT2FeaturesSegmenter(GPT2Segmenter):
    def get_uncertainties(self, phonemes):
        """Gets a variety of measures of prediction uncertainty at each point in the sequence,
            extracted from a GPT2 features model.

        Parameters
        ----------
        phonemes : list of str
            A list of phonemes.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        # First we get the GPT2Segmenter uncertainties by having the model predict the phonemes rather than features
        return_token_logits_tmp = self.model.return_token_logits
        self.model.return_token_logits = True
        uncertainties = super().get_uncertainties(phonemes)
        self.model.return_token_logits = False

        # Prepare token ids tensor
        token_ids = [self.boundary_token] + self.tokenizer.convert_tokens_to_ids(phonemes)
        long_input_ids = torch.tensor(token_ids, dtype=torch.long).to(DEFAULT_DEVICE)
        logits = self.get_logits(long_input_ids)[:len(token_ids) - 1] # Remove the last prediction. May be longer due to padding.
        logits = logits.permute(1, 2, 0)

        # For this model, we get a loss per feature, per position
        loss_fct = CrossEntropyLoss(reduction="none")
        label_vectors = self.model.feature_map.as_indices(long_input_ids)[1:].long().T
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
