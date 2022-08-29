__author__ = 'Zeb Goriely'
""" Wrapper class for training and evaluating a phoneme transformer model """

import typing
import torch
import logging 
import time 
import os 

import math
import random
import hashlib

import numpy as np
import torch
import torch.optim as optim
import torch.onnx

from .data import data
from .model.model import next_char_transformer
from .utils import DEVICE as DEFAULT_DEVICE, num_gpus

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainingStats(object):
    """Stores training and validation loss over time"""
    def __init__(self):
        self.training_losses = []
        self.validation_losses = []
        self.best_validation_epoch = 0

    def save(self, path):
        with open(path, 'w') as save_path:
            save_path.write(f'Training losses: {self.training_losses}\n')
            save_path.write(f'Validation losses: {self.validation_losses}\n')
            save_path.write(f'Best validation epoch: {self.best_validation_epoch}')
class PhoneTransformer(object):
    """
    Orchestrates model loading, training and evaluation using a specific 
    the next char transformer.
    """
    
    def __init__(self, config):
        """ Initialize base model based on a config 
        """

        # config params need to be accessed by several methods
        self.config = config

        self.base_device = config.get('TRAINING', 'device', fallback=DEFAULT_DEVICE)
        logger.info(f'Running phone transformer on device: {self.base_device}')


    def __call__(self) -> None: 
        """ 
        Train or evaluate the model
        """

        ###############################################################################
        # Prepare experiment
        ###############################################################################

        log_interval = self.config.getint('LOGGING', 'log_interval', fallback=200)

        # TODO: Replace all file output with Weights and Biases
        exp_dir = self.config.get('EXPERIMENT', 'directory', fallback="")
        logging.info(f'Saving output and log file to {exp_dir}')

        model_path = os.path.join(exp_dir, 'model.pt')
        training_stats_path = os.path.join(exp_dir, 'training_stats.txt')

        ###############################################################################
        # Load data
        ###############################################################################

        data_dir = self.config.get('DATASET', 'root_path', fallback="")
        sequence_length = self.config.getint('TRAINING', 'sequence_length')
        if not data_dir:
            logger.exception('No training data specified.')
            raise Exception('No training data specified')

        fn = 'corpus.{}.data'.format(hashlib.md5(data_dir.encode()).hexdigest())
        if os.path.exists(fn):
            logging.info('Loading cached dataset...')
            corpus = torch.load(fn)
        else:
            logging.info('Producing dataset...')
            corpus = data.Corpus(data_dir, sequence_length, False)
            torch.save(corpus, fn)

        # Starting from sequential data, batchify arranges the dataset into columns.
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

        def batchify(data, batch_size):
            # Work out how cleanly we can divide the dataset into batch_size parts.
            # Also ensure that each batch starts at the start of an utterance
            nbatch = data.size(0) // (batch_size * sequence_length)
            # Trim off any extra elements that wouldn't cleanly fit (remainders).
            data = data.narrow(0, 0, nbatch * batch_size * sequence_length)
            # Evenly divide the data across the batch_size batches.
            data = data.view(batch_size, -1).t().contiguous()
            return data.to(self.base_device)


        pad = 0
        assert pad == corpus.dictionary.word2idx["<pad>"]

        batch_size = self.config.getint('TRAINING', 'batch_size')
        logging.info(f'Using batch_size = {batch_size}')
        eval_batch_size = batch_size
        test_batch_size = 1
        train_data = batchify(corpus.train, batch_size)
        val_data = batchify(corpus.valid, eval_batch_size)
        test_data = batchify(corpus.test, test_batch_size)

        ###############################################################################
        # Build the model
        ###############################################################################

        logging.info('Building model...')
        vocab_size = len(corpus.dictionary)
        hidden_size = self.config.getint('MODEL', 'hidden_size', fallback=64)
        n_layers = self.config.getint('MODEL', 'n_layers', fallback=16)
        dropout = self.config.getfloat('MODEL', 'dropout', fallback=0.1)
        tied = self.config.getboolean('MODEL', 'tied', fallback=False)
        model = next_char_transformer(vocab_size, hidden_size=hidden_size, n_layers=n_layers,
                                    dropout=dropout, tied=tied, max_sequence_len=sequence_length,
                                    intermediate_losses=True).to(self.base_device)


        ###############################################################################
        # Training code
        ###############################################################################

        # mask subsequent entries
        def subsequent_mask(size):
            """Mask out subsequent positions."""
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0


        def make_std_mask(tgt):
            """Create a mask to hide padding and future words."""
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
            return tgt_mask


        # get_batch subdivides the source data into chunks of length args.bptt.
        # If source is equal to the example output of the batchify function, with
        # a bptt-limit of 2, we'd get the following two Variables for i = 0:
        # ┌ a g m s ┐ ┌ b h n t ┐
        # └ b h n t ┘ └ c i o u ┘
        # Note that despite the name of the function, the subdivison of data is not
        # done along the batch dimension (i.e. dimension 1), since that was handled
        # by the batchify function. The chunks are along dimension 0, corresponding
        # to the seq_len dimension in the LSTM.

        def get_batch(source, i, train):
            if train:
                #i = torch.randint(low=0, high=(len(source) - args.bptt), size=(1,)).long().item()
                i = torch.randint(low=0, high=(len(source) // sequence_length), size=(1,)).long().item() * sequence_length
            #else:
                # seq_len = min(args.bptt, len(source) - 1 - i)
                # target = source[i + seq_len, :]
                # target = source[i + 1:i + 1 + seq_len].t()

            seq_len = min(sequence_length, len(source) - 1 - i)
            target = source[i + 1:i + 1 + seq_len].t()
            data = source[i:i + seq_len].t()

            data_mask = (data != pad).unsqueeze(-2)
            target_mask = make_std_mask(data.long())

            # reshape target to match what cross_entropy expects
            target = target.contiguous().view(-1)

            return data, target, data_mask, target_mask


        def evaluate(data_source):
            # Turn on evaluation mode which disables dropout.
            total_loss = AverageMeter()
            model.eval()
            ntokens = len(corpus.dictionary)
            step = 1
            with torch.no_grad():
                # Original code slid a window along of size 1, rather than per-utterance
                #for batch, i in enumerate(range(0, data_source.size(0) - 1 - args.bptt, step)):
                for batch, i in enumerate(range(0, data_source.size(0) - 1, sequence_length)):
                    data, target, data_mask, target_mask = get_batch(data_source, i, train=False)
                    output = model(data, target_mask)
                    _, last_loss = model.criterion(output, target)
                    total_loss.update(last_loss.item(), data.size(0))
            return total_loss.avg

        def train():
            # Turn on training mode which enables dropout.
            model.train()
            total_loss = AverageMeter()
            start_time = time.time()
            ntokens = len(corpus.dictionary)
            for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):
                data, target, data_mask, target_mask = get_batch(train_data, i, train=True)
                model.zero_grad()
                output = model(data, target_mask)
                loss, last_loss = model.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss.update(last_loss.item(), data.size(0))

                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss.avg
                    elapsed = time.time() - start_time
                    logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        epoch, batch, len(train_data) // sequence_length,
                        elapsed * 1000 / log_interval, cur_loss,
                        math.exp(cur_loss), cur_loss / math.log(2)))
                    total_loss.reset()
                    start_time = time.time()

                if batch % 10000 == 0 and batch > 0:
                    break

            return total_loss.avg

        # Loop over epochs.
        best_val_loss = None
        lr = self.config.getfloat('TRAINING', 'lr', fallback=0.003)
        momentum = self.config.getfloat('TRAINING', 'momentum', fallback=0.99)
        optimizer = optim.SGD(model.parameters(), lr, momentum)

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        logging.info('Number of parameters: {}'.format(num_params))

        epochs = self.config.getint('TRAINING', 'epochs', fallback=2)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            stats = TrainingStats()
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train_loss = train()
                stats.training_losses.append(train_loss)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                    'train ppl {:8.2f} | train bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                                train_loss, math.exp(train_loss), train_loss / math.log(2)))
                val_loss = evaluate(val_data)
                stats.validation_losses.append(val_loss)
                logging.info('-' * 89)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss), val_loss / math.log(2)))
                logging.info('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(model_path, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                    stats.best_validation_epoch = epoch
                model.update(epoch // epochs)

        except KeyboardInterrupt:
            logging.info('-' * 89)
            logging.info('Exiting from training early')

        # Save training stats
        stats.save(training_stats_path)
        logging.info(f'Saved training stats to {training_stats_path}')

        # Load the best saved model.
        with open(model_path, 'rb') as f:
            logging.info(f'Loading best model from {model_path}')
            model = torch.load(f)

        # Run on test data.
        test_loss = evaluate(test_data)
        logging.info('=' * 89)
        logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        logging.info('=' * 89)
