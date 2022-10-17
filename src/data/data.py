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
    """ Pre-processes and stores a corpus of train, test and validation data. 

    Given a path, expects to find a "train.txt", "valid.txt" and "test.txt" file at that path. 
    For each file, tokenises each line in that file, truncating it if it exceeds the maximum
    utterance length and padding it otherwise. Each split of the data is stored as a single
    long tensor of token IDs.

    Expects space-delimited characters, but can also tokenize raw text if `raw_text` is true. In
    this case, lines are not truncated.
    """

    def __init__(self, path, max_utterance_length=64, truncate_long_utterances=False, raw_text=False):
        self.max_utterance_length = max_utterance_length
        self.truncate_long_utterances = truncate_long_utterances
        self.raw_text = raw_text

        self.dictionary = Dictionary()
        if not raw_text:
            self.dictionary.add_word('<PAD>')
            self.dictionary.add_word('<START>')
            self.dictionary.add_word('<END>')
            assert PAD == self.dictionary.word2idx["<PAD>"]

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize_raw(self, path):
        """Tokenizes a raw text file, converting all characters to IDs.
        Performs no padding and adds no EOS tokens.
        """
        if not os.path.exists(path):
            logger.exception(f'No text file found at {path}')
            raise Exception(f'No text file found at {path}')

        # Add tokens to the dictionary
        with open(path, 'r') as f:
            lines = f.readlines()
            tensor_length = sum([len(line) for line in lines])
            logging.info(f'Found {tensor_length} characters in {path}')
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
        

    def tokenize_line(self, line):
        """ Splits, pads and adds EOS to a line. Returns tokenized line and whether line was truncated.
        Expects a space-delimited line, such as "h e l l o".
        """
        tokens = line.split()
        truncated = False
        if len(tokens) + 2 > self.max_utterance_length:
            truncated = True
            tokens = tokens[:self.max_utterance_length - 2]
        pad_length = self.max_utterance_length - len(tokens) - 2
        tokens = ['<START>'] + tokens + ['<END>'] + ['<PAD>'] * pad_length
        assert (len(tokens) == self.max_utterance_length)
        
        return tokens, truncated

    def tokenize_delimited(self, path):
        """Tokenizes a space delimited text file. Returns a single tensor containing all tokens as IDs. """
        if not os.path.exists(path):
            logger.exception(f'No text file found at {path}')
            raise Exception(f'No text file found at {path}')
        
        # Add tokens to the dictionary
        with open(path, 'r') as f:
            lines = f.readlines()
            tokenized_lines = []
            long_utterances = 0
            for line in lines:
                tokens, truncated = self.tokenize_line(line)
                if truncated:
                    long_utterances += 1
                    if not self.truncate_long_utterances:
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
        
        logging.info(f'Found {len(lines)} utterances in {path}')
        if long_utterances > 0:
            if self.truncate_long_utterances:
                logging.info(f'Truncated {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
            else:
                logging.info(f'Discarded {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
        logging.info(f'Saved {len(tokenized_lines)} utterances')

        return ids

    def tokenize(self, path):
        """ Either tokenizes a space delimited file or a raw text file"""
        if self.raw_text:
            return self.tokenize_raw(path)
        else:
            return self.tokenize_delimited(path)

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
        self.batch_size = batch_size

        # Work out how cleanly we can divide the dataset into batch_size parts.
        # Also ensure that each batch starts at the start of an utterance
        num_batches = data.size(0) // (batch_size * self.sequence_length)
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, num_batches * batch_size * self.sequence_length)
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
            # Choose a random sequence in the data if we're training, otherwise use given batch number
            i = torch.randint(low=0, high=(len(self.data) // self.sequence_length), size=(1,)).long().item() * self.sequence_length
        
        # Make sure we don't spill over the edge of the data
        seq_len = min(self.sequence_length, len(self.data) - 1 - i)
        data = self.data[i:i + seq_len].t()
        # Target is sequence shifted by 1.
        target = self.data[i + 1:i + 1 + seq_len].t()
        # Ensure last token is pad
        target[:,-1] = torch.zeros(self.batch_size)

        # Mask out pad token
        data_mask = (data != PAD).unsqueeze(-2)
        target_mask = make_std_mask(data.long())

        # reshape target to match what cross_entropy expects
        target = target.contiguous().view(-1)

        return data, target, data_mask, target_mask
