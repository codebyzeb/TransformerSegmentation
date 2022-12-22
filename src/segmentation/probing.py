import logging
import os
import time
from typing import Any
import yaml
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')

from src.data.data import subsequent_mask
from .evaluate import evaluate
from src.model.model import next_char_transformer

class WordBoundaryDataset(Dataset):
    """ Dataset for word boundary prediction. """

    def __init__(self, utterance_file, corpus, single_utterance=False, max_sequence_length=128):
        self.corpus = corpus
        self.max_sequence_length = max_sequence_length
        self.single_utterance = single_utterance

        gold_utterances = [line.strip() for line in open(utterance_file, 'r')
            if len([phone for phone in line.strip(' ') if phone!=';eword']) <= max_sequence_length-1][:1000]
        
        boundaries = []
        utterances = []
        self.utterance_starts = []
        tensor_length = 0
        for utterance in gold_utterances:
            self.utterance_starts.append(tensor_length)
            processed = utterance.split(' ')
            phonemes = [corpus.dictionary.word2idx['<BOUNDARY>']]
            word_starts = [0]
            next_word = 1
            for c in processed:
                if c == ';eword':
                    next_word = 1
                else:
                    phonemes.append(corpus.dictionary.word2idx[c])
                    word_starts.append(next_word)
                    next_word = 0
            utterances.extend(phonemes)
            boundaries.extend(word_starts)
            tensor_length += len(phonemes)

        self.boundaries = torch.LongTensor(boundaries)
        self.utterances = torch.LongTensor(utterances)
        
    def __len__(self):
        """ Return the number of utterances or the number of sequences. """
        return len(self.utterance_starts) if self.single_utterance else len(self.utterances) // self.max_sequence_length

    def __getitem__(self, idx):
        """ Return the utterance or sequence at the given index. """
        if self.single_utterance:
            utterance_start = self.utterance_starts[idx]
            utterance_end = self.utterance_starts[idx+1] if idx < len(self.utterance_starts)-1 else len(self.utterances)
            return self.utterances[utterance_start:utterance_end], self.boundaries[utterance_start:utterance_end]
        else:
            return self.utterances[idx*self.max_sequence_length:(idx+1)*self.max_sequence_length], self.boundaries[idx*self.max_sequence_length:(idx+1)*self.max_sequence_length]

class BoundaryProbe(nn.Module):
    """ A word boundary probe for the next character prediction task. """

    def __init__(self, model):
        """ Initiate layers for the probe. 
        Args:
            model: The next character prediction model    
        """
        super(BoundaryProbe, self).__init__()

        self.model = model
        self.hidden_size = model.hidden_size
        self.classifier_layer = nn.Linear(self.hidden_size , 2)
        
    def forward(self, src, mask):
        """ Take in and process masked src and target sequences. 
        Args:
            src: The source sequence
            mask: The mask for the source sequence
        Returns:
            The output of the classifier layer
        """
        src_emb = self.model.embed(src)
        emb, _ = self.model.encoder(src_emb, mask)
        x = self.classifier_layer(emb)
        return x

class ProbeTrainer(object):
    """ Orchestrates probe loading, training and evaluation using the next char transformer. """
    
    def __init__(self, config_file, checkpoint_path, train_data, test_data, seed=32):
        # Set the random seed manually for reproducibility.
        torch.manual_seed(seed)

        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.DEBUG,
            handlers=[logging.StreamHandler()]
        )

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Loading on device: {self.device}')

        # Load config
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        logging.info(f'Loaded config: {self.config}')
        
        # Load corpus
        data_dir = self.config['root_path']['value']
        fn = 'corpus.{}.data'.format('.'.join(data_dir.split('/')))
        if os.path.exists(fn):
            logging.info('Loading cached dataset...')
            self.corpus = torch.load(fn)
            ntokens = len(self.corpus.dictionary)
        else:
            logging.info('No precached dataset found')
            raise Exception('No precached dataset found')

        # Create Datasets
        training_data = WordBoundaryDataset(train_data, self.corpus, single_utterance=False, max_sequence_length=64)
        test_data = WordBoundaryDataset(test_data, self.corpus, single_utterance=True, max_sequence_length=self.config['sequence_length']['value'])

        self.train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

        # Load model
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device)
            model = next_char_transformer(ntokens,
                                            n_layers=self.config['n_layers']['value'],
                                            hidden_size=self.config['hidden_size']['value'],
                                            inner_linear=self.config['inner_linear']['value'],
                                            max_sequence_len=self.config['sequence_length']['value']).to(self.device)
        model.load_state_dict(checkpoint['learner_state_dict'], strict=False)
        model.eval()

        # Create probe and optimizer
        self.probe = BoundaryProbe(model).to(self.device)
        self.optimizer = optim.SGD(self.probe.classifier_layer.parameters(), lr=0.001, momentum=0.9)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        num_epochs = kwds.get('epochs', 20)
        length = len(self.train_dataloader)
        print(length)
        best_accuracy = 0

        for epoch in range(num_epochs):
    
            i = 0
            self.probe.train()
            self.probe.model.eval()
            for phonemes, boundaries in iter(self.train_dataloader):
                phonemes = phonemes.to(self.device)
                boundaries = boundaries.to(self.device)
                self.optimizer.zero_grad()
                mask = subsequent_mask(phonemes.shape[1])
                outputs = self.probe(phonemes, mask)
                loss = F.cross_entropy(outputs.view(-1, 2), boundaries.view(-1))
                loss.backward()
                self.optimizer.step()
                i+=1
                if i % 100 == 0:
                    logging.info('Epoch: %d, Loss: %f, Batch: %d/%d' % (epoch, loss.item(), i, length))
        
            test_error_count = 0.0
            total_boundaries = 1
            gold_utterances = []
            predicted_utterances = []
            self.probe.eval()
            for phonemes, boundaries in iter(self.test_dataloader):
                phonemes = phonemes.to(self.device)
                boundaries = boundaries.to(self.device)
                mask = subsequent_mask(phonemes.shape[1])
                outputs = self.probe(phonemes, mask)
                test_error_count += float(torch.sum(torch.abs(boundaries.view(-1) - outputs.view(-1, 2).argmax(1))))
                total_boundaries += outputs.shape[0] * outputs.shape[1]
                for i, utterance in enumerate(phonemes):
                    predicted_boundaries = outputs[i].argmax(1)
                    gold_utterances.append(' '.join([(';eword ' if b.item() else '') + self.corpus.dictionary.idx2word[c.item()] for c, b in zip(utterance[1:], boundaries[i,1:])]))
                    predicted_utterances.append(' '.join([(';eword ' if b.item() else '') + self.corpus.dictionary.idx2word[c.item()] for c, b in zip(utterance[1:], predicted_boundaries[1:])]))
        
            results = evaluate(gold_utterances, predicted_utterances)
            test_accuracy = 1.0 - float(test_error_count) / total_boundaries
            logging.info('Test Accuracy for Epoch %d: %f' % (epoch, test_accuracy))
            logging.info('Segmentation results:')
            logging.info(results)
            if test_accuracy > best_accuracy:
                # torch.save(model.state_dict(), BEST_MODEL_PATH)
                best_accuracy = test_accuracy