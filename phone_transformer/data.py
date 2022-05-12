import os
import torch

from collections import Counter


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
    def __init__(self, path, max_utterance_length=64, truncate_long_utterances=False):
        self.max_utterance_length = max_utterance_length
        self.truncate_long_utterances = truncate_long_utterances

        self.dictionary = Dictionary()
        self.dictionary.add_word('<pad>')
        self.dictionary.add_word('<ub>')

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
        
        print(f'Found {len(lines)} utterances in {path}')
        if long_utterances > 0:
            if self.truncate_long_utterances:
                print(f'Truncated {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
            else:
                print(f'Discarded {long_utterances} utterances that were longer than max sequence length of {self.max_utterance_length}')
        print(f'Saved {len(tokenized_lines)} utterances')

        return ids
