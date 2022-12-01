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

class RawTokenizer(object):
    """ Creates a dictionary, tokenizes raw text and returns a tensor of token ID.
        Converts all characters to IDs, performing no padding and adding no EOS tokens. """

    def __init__(self):
        self.dictionary = Dictionary()
        # We always add the padding token, even if it isn't used, because other functions mask it
        self.dictionary.add_word('<PAD>')

    def tokenize(self, path):
        if not os.path.exists(path):
            logger.exception(f'No text file found at {path}')
            raise Exception(f'No text file found at {path}')
        logger.info(f'Loading in characters from {path}')
        with open(path, 'r') as f:
            lines = f.readlines()
        return self.tokenize_lines(lines)

    def tokenize_lines(self, lines):
        logger.info('Tokenizing raw text (including spaces)')
        tensor_length = sum([len(line) for line in lines])
        logging.info(f'Found {tensor_length} characters in file')
        token_id = 0
        ids = torch.LongTensor(tensor_length)
        for line in lines:
            for token in line:
                self.dictionary.add_word(token)
                ids[token_id] = self.dictionary.word2idx[token]
                token_id += 1
        
        assert token_id == tensor_length

        logging.info(f'Saved {tensor_length} characters')

        return ids

class SpaceTokenizer(RawTokenizer):
    """ Creates a dictionary, tokenizes text according to space characters and returns a tensor of token ID.
        Converts all characters to IDs and additionally, adds a <BOUNDARY> token between each line in the file. Removes banned tokens. """

    def __init__(self, banned_tokens=None):
        RawTokenizer.__init__(self)
        self.dictionary.add_word('<BOUNDARY>')
        self.banned_tokens = banned_tokens

    def remove_tokens(self, lines):
        if self.banned_tokens:
            new_lines = []
            for line in lines:
                for token in self.banned_tokens:
                    line = [t for t in line if t != token]
                new_lines.append(line)
            lines = new_lines
        return lines

    def tokenize_lines(self, lines):
        logger.info('Tokenizing text using space character, merging utterances')
        lines = [line.split() + ['<BOUNDARY>'] for line in lines]
        lines = self.remove_tokens(lines)
                    
        tensor_length = sum([len(line) for line in lines])
        logging.info(f'Found {tensor_length-len(lines)} characters in file')
        ids = torch.LongTensor(tensor_length)
        token_id = 0
        for line in lines:
            for token in line:
                self.dictionary.add_word(token)
                ids[token_id] = self.dictionary.word2idx[token]
                token_id += 1
            
        assert token_id == tensor_length
        
        logging.info(f'Added {len(lines)} utterance boundaries')
        logging.info(f'Saved {tensor_length} total characters')

        return ids

class TruncateTokenizer(RawTokenizer):
    """ Creates a dictionary, tokenizes text according to space characters and preserves line boundaries by 
        truncating and adding start, end and padding tokens to each line. Converts all characters to IDs. Removes banned tokens. """

    def __init__(self, max_utterance_length, banned_tokens=None):
        RawTokenizer.__init__(self)
        self.max_utterance_length = max_utterance_length
        self.dictionary.add_word('<START>')
        self.dictionary.add_word('<END>')
        self.banned_tokens = banned_tokens

    def split_and_pad_line(self, line):
        """ Splits, pads and adds EOS to a line. Returns tokenized line and whether line was truncated.
            Expects a space-delimited line, such as "h e l l o". Removes banned tokens. """
        tokens = line.split()
        if self.banned_tokens:
            for token in self.banned_tokens:
                if token in line:
                    line = [t for t in line if t != token]
        truncated = False
        if len(tokens) + 2 > self.max_utterance_length:
            truncated = True
            tokens = tokens[:self.max_utterance_length - 2]
        pad_length = self.max_utterance_length - len(tokens) - 2
        tokens = ['<START>'] + tokens + ['<END>'] + ['<PAD>'] * pad_length
        assert (len(tokens) == self.max_utterance_length)
        
        return tokens, truncated

    def tokenize_lines(self, lines):
        logger.info('Tokenizing text using space character, keeping utterances separate by adding padding') 
        tokenized_lines = []
        long_utterances = 0
        for line in lines:
            tokens, truncated = self.split_and_pad_line(line)
            if truncated:
                long_utterances += 1
                continue
            tokenized_lines.append(tokens)
            for token in tokens:
                self.dictionary.add_word(token)
        
        tensor_length = len(tokenized_lines) * self.max_utterance_length
        ids = torch.LongTensor(tensor_length)
        token_id = 0
        for line in tokenized_lines:
            for token in line:
                ids[token_id] = self.dictionary.word2idx[token]
                token_id += 1
        
        assert token_id == len(tokenized_lines) * self.max_utterance_length
        
        logging.info(f'Found {len(lines)} utterances in file')
        if long_utterances > 0:
            logging.info(f'Discarded {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
        logging.info(f'Saved {len(tokenized_lines)} utterances')

        return ids


class Corpus(object):
    """ Pre-processes and stores a corpus of train, test and validation data. 

    Given a path, expects to find a "train.txt", "valid.txt" and "test.txt" file at that path. 
    For each file, tokenises each line in that file, truncating it if it exceeds the maximum
    utterance length and padding it otherwise. Each split of the data is stored as a single
    long tensor of token IDs.

    """

    def __init__(self, path, tokenizer):
        self.dictionary = tokenizer.dictionary
        self.tokenizer = tokenizer

        self.train = self.tokenizer.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenizer.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenizer.tokenize(os.path.join(path, 'test.txt'))

# mask subsequent entries
def subsequent_mask(size):
    """ Mask out subsequent positions. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt):
    """ Create a mask to hide padding and future words. """
    tgt_mask = (tgt != PAD).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask

class BatchedData():
    """ Starting from sequential data, batches it into columns for efficient processing. """

    # Starting from sequential data, arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and `batch_size` 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def __init__(self, data, batch_size, sequence_length, base_device):
        self.sequence_length = sequence_length
        self.base_device = base_device
        self.batch_size = batch_size

        # Make sure every batch is aligned with the start of an utterance
        num_batches = data.size(0) // (batch_size * self.sequence_length)
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, num_batches * batch_size * self.sequence_length)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data.to(self.base_device)

    # get_batch subdivides the data into chunks of length `sequence_length`.
    # If source is equal to the example output of the batchify function, with
    # a sequence_length of 2, we'd get the following for data, target for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the sequence_length dimension in the LSTM.

    def get_batch(self, i):        
        # Make sure we don't spill over the edge of the data
        seq_len = min(self.sequence_length, len(self.data) - 1 - i)
        data = self.data[i:i + seq_len].t()

        # Target is sequence shifted by 1.
        target = self.data[i + 1:i + 1 + seq_len].t()

        # Mask out pad token
        data_mask = (data != PAD).unsqueeze(-2)
        target_mask = make_std_mask(data.long())

        # reshape target to match what cross_entropy expects
        target = target.contiguous().view(-1)

        return data, target, data_mask, target_mask
