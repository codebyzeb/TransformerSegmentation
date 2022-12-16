""" Methods for segmenting utterances using a transformer model's phoneme predictions """

import logging
import numpy as np
import pandas as pd
import scipy
import torch
import sys

sys.path.append('../')
from src.data.data import subsequent_mask

from .evaluate import evaluate

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def segment_by_cutoff(utterance, measure, cutoff):
    """ Segments an utterance by a given measure and cutoff value.
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

    segmented_utterance = ' '.join(';eword ' + p if m > cutoff else p for p,m in zip(utterance.Phoneme, utterance[measure])).strip()
    return segmented_utterance
    
def segment_by_spike(data, measure):
    """ Segments an utterance by a given measure whenever there's a spike in the measure.
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

    before = np.delete(np.pad(data[measure], (1,0)), len(data))
    after = np.delete(np.pad(data[measure], (0,1)), 0)
    boundaries = np.logical_and(data[measure] > before, data[measure] > after)
    segmented_utterance = ' '.join(';eword ' + p if b else p for p,b in zip(data.Phoneme, boundaries)).strip()
    return segmented_utterance

class Segmenter(object):
    
    def __init__(self, model, utterances, corpus):
        """ A class for segmenting utterances using a transformer model's phoneme predictions.
        Parameters
        ----------
        model : torch.nn.Module
            A transformer model.
        utterances : list
            A list of utterances to segment.
        corpus : Corpus
            A Corpus object containing the dictionary used by the model.
        """

        self.max_sequence_length = 128
        self.model = model
        self.corpus = corpus
        self.boundary_token = self.corpus.dictionary.word2idx['<BOUNDARY>']

        self.gold_utterances = [line.strip() for line in utterances 
            if len([phone for phone in line.strip(' ') if phone!=';eword']) <= self.max_sequence_length-1]

        self.processed_utterances = []
        for utt in self.gold_utterances:
            self.processed_utterances.append(self.process_utterance(utt))

    def process_utterance(self, utterance):
        """ Processes an utterance into a dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
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

        processed = utterance.split(' ')
        phonemes = []
        word_starts = []
        next_word = True
        for c in processed:
            if c == ';eword':
                next_word = True
            else:
                phonemes.append(c)
                word_starts.append(next_word)
                next_word = False

        data = self.get_uncertainties(['<BOUNDARY>'] + phonemes) # Might want to add <START> token if using TruncateTokenizer
        data['Pos'] = list(range(len(word_starts)))
        data['Starts'] = word_starts
        data['Phoneme'] = phonemes
        return pd.DataFrame(data)
        
    def get_uncertainties(self, utterance):
        """ Gets a variety of measures of prediction uncertainty at each point in the sequence.
        Parameters
        ----------
        utterance : list
            A list of phones.

        Returns
        -------
        uncertainties : dict
            A dictionary containing a variety of measures of prediction uncertainty at each point in the sequence.
        """

        token_ids = [self.corpus.dictionary.word2idx[t] for t in list(utterance)]
        uncertainties = {'Entropy' : [], 'Increase in Entropy' : [],
                        'Loss' : [], 'Increase in Loss' : [],
                        'Rank' : [], 'Increase in Rank' : [],
                        'Boundary Prediction' : [], 'Increase in Boundary Prediction' : []}
        entropy = 0
        boundary_prediction = 0
        loss = 0
        rank = 0

        with torch.no_grad():
            input = torch.LongTensor([token_ids], device=DEFAULT_DEVICE)
            mask = subsequent_mask(len(token_ids))
            output = self.model(input,mask)[-1]

            for i in range(len(token_ids)-1):
                token_predictions = torch.exp(output[0][i])
                
                new_entropy = scipy.stats.entropy(token_predictions, base=2)
                increase_in_entropy = new_entropy - entropy
                entropy = new_entropy

                log_token_predictions = torch.tensor([list(output[0][i])], dtype=torch.float, requires_grad=True)
                target = torch.tensor([token_ids[i+1]], dtype=torch.long)
                new_loss = torch.nn.functional.cross_entropy(log_token_predictions, target)
                increase_in_loss = new_loss - loss
                loss = new_loss

                new_rank = np.log2(1+(token_predictions.argsort(descending=True) == token_ids[i+1]).nonzero(as_tuple=True)[0])
                increase_in_rank = new_rank - rank
                rank = new_rank

                new_boundary_prediction = token_predictions[self.boundary_token]
                increase_in_boundary_prediction = new_boundary_prediction - boundary_prediction
                boundary_prediction = new_boundary_prediction

                uncertainties['Entropy'].append(entropy)
                uncertainties['Increase in Entropy'].append(increase_in_entropy)
                uncertainties['Loss'].append(loss)
                uncertainties['Increase in Loss'].append(increase_in_loss)
                uncertainties['Rank'].append(rank)
                uncertainties['Increase in Rank'].append(increase_in_rank)
                uncertainties['Boundary Prediction'].append(boundary_prediction)
                uncertainties['Increase in Boundary Prediction'].append(increase_in_boundary_prediction)
                
        return uncertainties

    def evaluate_cutoff_segmentation(self, measure, cutoffs):
        """ Given a measure and a range of cutoffs, segments according to the cutoff and evaluates the results.
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
            results = evaluate(segmented_utterances, self.gold_utterances)
            results['Cutoff'] = cutoff
            all_results.append(results)

        logging.info('Segmented utterances for a range of cutoffs using measure: {}'.format(measure))
        return pd.DataFrame(all_results)

    def evaluate_spike_segmentation(self, measure):
        """ Given a measure, segments according to spikes in the measure and evaluates the results.
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

        logging.info('Segmented utterances according to spikes in measure: {}'.format(measure))
        return evaluate(segmented_utterances, self.gold_utterances)
        