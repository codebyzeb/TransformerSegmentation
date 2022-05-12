# coding: utf-8
import argparse
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

import data
from transformer_model import next_char_transformer

parser = argparse.ArgumentParser(description='Phoneme-level Transformer Language Model')
parser.add_argument('--data', type=str, default='./experiment/EnglishNA_debug/prepared',
                    help='location of the data corpus')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--n_layers', type=int, default=64,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.003,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.99,
                    help='momentum for SGD')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=250,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=64,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.55,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model_training',
                    help='folder to save the final model and training info')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Prepare output folder
###############################################################################


if not os.path.exists(args.save):
    print(f'Creating directory {args.save} to save output')
    os.mkdir(args.save)
args_string = '\n'.join(str(args)[10:-1].split(', '))
print('Running with arguments:\n' + args_string)
args_filename = os.path.join(args.save, 'args.txt')
with open(args_filename, 'w') as args_file:
    args_file.write(args_string)
    print(f'Saved arguments to {args_filename}')

model_path = os.path.join(args.save, 'model.pt')
training_stats_path = os.path.join(args.save, 'training_stats.txt')

###############################################################################
# Load data
###############################################################################


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

class TrainingStats():
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

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data, args.bptt, False)
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
    nbatch = data.size(0) // (batch_size * args.bptt)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size * args.bptt)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


pad = 0
assert pad == corpus.dictionary.word2idx["<pad>"]

eval_batch_size = args.batch_size
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)

###############################################################################
# Build the model
###############################################################################

print('Building model...')
vocab_size = len(corpus.dictionary)
model = next_char_transformer(vocab_size, hidden_size=args.hidden_size, n_layers=args.n_layers,
                              dropout=args.dropout, tied=args.tied, max_sequence_len=args.bptt,
                              intermediate_losses=True).to(device)


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
        i = torch.randint(low=0, high=(len(source) // args.bptt), size=(1,)).long().item() * args.bptt
    #else:
        # seq_len = min(args.bptt, len(source) - 1 - i)
        # target = source[i + seq_len, :]
        # target = source[i + 1:i + 1 + seq_len].t()

    seq_len = min(args.bptt, len(source) - 1 - i)
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
        for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
            data, target, data_mask, target_mask = get_batch(data_source, i, train=False)
            output = model(data, target_mask)
            _, last_loss = model.criterion(output, target)
            total_loss.update(last_loss.item(), data.size(0))
    return total_loss.avg

def printer(data):
    return [corpus.dictionary.idx2word[id] for id in data]

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = AverageMeter()
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, target, data_mask, target_mask = get_batch(train_data, i, train=True)
        model.zero_grad()
        output = model(data, target_mask)
        loss, last_loss = model.criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss.update(last_loss.item(), data.size(0))

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.avg
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt,
                elapsed * 1000 / args.log_interval, cur_loss,
                math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss.reset()
            start_time = time.time()

        if batch % 10000 == 0 and batch > 0:
            break

    return total_loss.avg

# Loop over epochs.
best_val_loss = None
optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)

num_params = 0
for p in model.parameters():
    num_params += p.numel()

print('Number of parameters: {}'.format(num_params))


# At any point you can hit Ctrl + C to break out of training early.
try:
    stats = TrainingStats()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        stats.training_losses.append(train_loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
              'train ppl {:8.2f} | train bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                         train_loss, math.exp(train_loss), train_loss / math.log(2)))
        val_loss = evaluate(val_data)
        stats.validation_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            stats.best_validation_epoch = epoch
        model.update(epoch // args.epochs)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Save training stats
stats.save(training_stats_path)
print(f'Saved training stats to {training_stats_path}')

# Load the best saved model.
with open(model_path, 'rb') as f:
    print(f'Loading best model from {model_path}')
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
