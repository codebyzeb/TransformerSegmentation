import os
import torch
import logging

import numpy as np

from collections import Counter

logger = logging.getLogger(__name__)

PAD = 0

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """ Stores a corpus, split according to train-valid-test. Sequences have a maximum length
    and are stored as IDs rather than word tokens.
    """

    def __init__(self, path, max_utterance_length=64, truncate_long_utterances=False):
        self.max_utterance_length = max_utterance_length
        self.truncate_long_utterances = truncate_long_utterances

        self.dictionary = Dictionary()
        self.dictionary.add_word('<pad>')
        self.dictionary.add_word('<ub>')
        assert PAD == self.dictionary.word2idx["<pad>"]

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize_line(self, line):
        """Splits, pads and adds EOS to a line. Returns tokenized line and whether line was truncated."""
        words = line.split()
        truncated = False
        if len(words) + 2 > self.max_utterance_length:
            truncated = True
            words = words[:self.max_utterance_length - 2]
        pad_length = self.max_utterance_length - len(words) - 2
        words = ['<ub>'] + words + ['<ub>'] + ['<pad>'] * pad_length
        assert (len(words) == self.max_utterance_length)
        
        return words, truncated

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary

        with open(path, 'r') as f:
            lines = f.readlines()
            tokenized_lines = []
            long_utterances = 0
            for line in lines:
                words, truncated = self.tokenize_line(line)
                if truncated:
                    long_utterances += 1
                    if not self.truncate_long_utterances:
                        continue
                tokenized_lines.append(words)
                for word in words:
                    self.dictionary.add_word(word)
            
            ids = torch.LongTensor(len(tokenized_lines) * self.max_utterance_length)
            token = 0
            for words in tokenized_lines:
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
            
            assert token == len(tokenized_lines) * self.max_utterance_length
        
        logging.info(f'Found {len(lines)} utterances in {path}')
        if long_utterances > 0:
            if self.truncate_long_utterances:
                logging.info(f'Truncated {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
            else:
                logging.info(f'Discarded {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
        logging.info(f'Saved {len(tokenized_lines)} utterances')

        return ids

# mask subsequent entries
def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt):
    """Create a mask to hide padding and future words."""
    tgt_mask = (tgt != PAD).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask

class BatchedData():
    """ Starting from sequential data, batches it into columns for efficient processing. """

    # Starting from sequential data, arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def __init__(self, data, batch_size, sequence_length, base_device, is_train):
        self.sequence_length = sequence_length
        self.base_device = base_device
        self.is_train = is_train

        # Work out how cleanly we can divide the dataset into batch_size parts.
        # Also ensure that each batch starts at the start of an utterance
        nbatch = data.size(0) // (batch_size * self.sequence_length)
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size * self.sequence_length)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data.to(self.base_device)

    # get_batch subdivides the data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(self, i):
        if self.is_train:
            #i = torch.randint(low=0, high=(len(source) - args.bptt), size=(1,)).long().item()
            i = torch.randint(low=0, high=(len(self.data) // self.sequence_length), size=(1,)).long().item() * self.sequence_length
        #else:
            # seq_len = min(args.bptt, len(source) - 1 - i)
            # target = source[i + seq_len, :]
            # target = source[i + 1:i + 1 + seq_len].t()

        seq_len = min(self.sequence_length, len(self.data) - 1 - i)
        target = self.data[i + 1:i + 1 + seq_len].t()
        data = self.data[i:i + seq_len].t()

        data_mask = (data != PAD).unsqueeze(-2)
        target_mask = make_std_mask(data.long())

        # reshape target to match what cross_entropy expects
        target = target.contiguous().view(-1)

        return data, target, data_mask, target_mask
