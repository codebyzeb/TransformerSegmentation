""" Methods for segmenting utterances using a transformer model's phoneme predictions """

import logging
import numpy as np
import pandas as pd
import scipy
import torch

from .evaluate import evaluate

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def segment_by_cutoff(utterance, measure, cutoff):
    boundaries = utterance[measure] > cutoff
    segmented_utterance = ' '.join(';eword ' + p if b else p for p,b in zip(utterance.Phoneme, boundaries)).strip()
    return segmented_utterance
    
def segment_by_spike(data, measure):
    before = np.delete(np.pad(data[measure], (1,0)), len(data))
    after = np.delete(np.pad(data[measure], (0,1)), 0)
    boundaries = np.logical_and(data[measure] > before, data[measure] > after)
    segmented_utterance = ' '.join(';eword ' + p if b else p for p,b in zip(data.Phoneme, boundaries)).strip()
    return segmented_utterance

class Segmenter(object):
    
    def __init__(self, model, path, corpus):
        """ Creates a segmenter object from the utterances in a path.
        Processes all utterances found (may take some time) to calculate uncertainty measures for segmentation. """

        self.max_sequence_length = 128
        self.model = model
        self.corpus = corpus
        self.boundary_token = self.corpus.dictionary.word2idx['<END>'] #if self.corpus.truncate_long_utterances else self.corpus.dictionary.word2idx['<BOUNDARY>']

        with open(path, 'r') as f:
            self.gold_utterances = [
                line.strip() for line in f.readlines()
                    if len([phone for phone in line.strip(' ') if phone!=';eword']) <= self.max_sequence_length-1]

        self.processed_utterances = []
        logging.info('Loading utterances from {}...'.format(path))
        for utt in self.gold_utterances:
            self.processed_utterances.append(self.process_utterance(utt))
        logging.info('Finished processing utterances')

    def process_utterance(self, utterance):
        """ Given an utterance with word boundary information, store the utterance, word starts and various uncertainty measures from the model at each position in the utterance.
        Parameters
        ----------
        utterance : str
            A space-separated string of phones with ';eword' used to indicate word boundaries.
            E.g. 'w ʌ t ;eword dʒ ʌ s t ;eword h æ p ə n d ;eword d æ d i ;eword'.
        
        Returns
        -------
        data : pandas.DataFrame
            A dataframe containing each phoneme (Phoneme) and phoneme position (Pos), whether or not a phoneme is the start of a word (Starts),
            the prediction entropy at each point in the sequence (Entropy) and the increase in entropy at each positoin (Increase in Entropy).
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

        data = self.get_uncertainties(['<START>'] + phonemes)
        data['Pos'] = list(range(len(word_starts)))
        data['Starts'] = word_starts
        return pd.DataFrame(data)
        
    def get_uncertainties(self, utterance):
        """ Gets prediction uncertainty at each point in the sequence. """

        token_ids = [self.corpus.dictionary.word2idx[t] for t in list(utterance)]
        uncertainties = {'Entropy' : [], 'Increase in Entropy' : [],
                        'Loss' : [], 'Increase in Loss' : [],
                        'Boundary Prediction' : [], 'Increase in Boundary Prediction' : []}
        entropy = 0
        boundary_prediction = 0
        loss = 0

        with torch.no_grad():
            for i in range(len(token_ids)-1):
                input = torch.LongTensor([token_ids[:i+1]], device=DEFAULT_DEVICE)
                mask = torch.ones(1,1,i+1)
                output = self.model(input,mask)[-1]
                last_token = torch.exp(output[0][i])
                
                new_entropy = scipy.stats.entropy(last_token, base=2)
                increase_in_entropy = new_entropy - entropy
                entropy = new_entropy

                input = torch.tensor([list(output[0][i])], dtype=torch.float)
                target = torch.tensor([token_ids[i]], dtype=torch.long)
                new_loss = torch.nn.functional.cross_entropy(input, target)
                increase_in_loss = new_loss - loss
                loss = new_loss

                new_boundary_prediction = last_token[self.boundary_token]
                increase_in_boundary_prediction = new_boundary_prediction - boundary_prediction
                boundary_prediction = new_boundary_prediction

                uncertainties['Entropy'].append(entropy)
                uncertainties['Increase in Entropy'].append(increase_in_entropy)
                uncertainties['Loss'].append(loss)
                uncertainties['Increase in Loss'].append(increase_in_loss)
                uncertainties['Boundary Prediction'].append(boundary_prediction)
                uncertainties['Increase in Boundary Prediction'].append(increase_in_boundary_prediction)
                
        return uncertainties

    def evaluate_cutoff_segmentation(self, measure, cutoffs):
        """ Given a measure and a range of cutoff values, segments according to those cutoffs and returns scores. """

        all_results = {}
        first = True
        for cutoff in cutoffs:
            segmented_utterances = [segment_by_cutoff(utt, measure, cutoff) for utt in self.processed_utterances]
            results = evaluate(segmented_utterances, self.gold_utterances)
            if first:
                first = False
                all_results = results
                for key in all_results:
                    all_results[key] = [all_results[key]]
                all_results['Cutoff'] = [cutoff]
            else:
                for key in all_results:
                    if key == 'Cutoff':
                        all_results[key].append(cutoff)
                    else:
                        all_results[key].append(results[key])

        logging.info('Segmented utterances for a range of cutoffs using measure: {}'.format(measure))
        return pd.DataFrame(all_results)

    def evaluate_spike_segmentation(self, measure):
        """ Given a measure, segments according to spikes in that measure. """
        segmented_utterances = [segment_by_spike(utt, measure) for utt in self.processed_utterances]
        logging.info('Segmented utterances according to spikes in measure: {}'.format(measure))
        return evaluate(segmented_utterances, self.gold_utterances)
