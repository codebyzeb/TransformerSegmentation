__author__ = 'Zeb Goriely'
""" Wrapper class for training and evaluating a phoneme transformer model """

import typing
import logging 
import time 
import os
import wandb

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

class PhoneTransformer(object):
    """
    Orchestrates model loading, training and evaluation using a specific 
    the next char transformer.
    """
    
    def __init__(self, config, resume_num_epochs=0):
        """ Initialize base model based on a config 
        """

        # config params need to be accessed by several methods
        self.config = config

        # whether to log out information to w&b
        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)

        self.base_device = config.get('TRAINING', 'device', fallback=DEFAULT_DEVICE)
        logging.info(f'Running phone transformer on device: {self.base_device}')

        # setting num_epochs before learner, to inform model if we are resuming training 
        # or starting fresh 
        self.num_epochs = resume_num_epochs if resume_num_epochs else 0

        # Load corpus, model and optimiser
        self.sequence_length = self.config.getint('TRAINING', 'sequence_length')
        self.corpus = self.load_corpus()
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()

        # NOTE: possibly load in learner checkpoint
        # if num_epochs is 0 at the start of training, then we are resuming training 
        if self.num_epochs > 0:
            checkpoint_file = "latest-checkpoint.pt"
            checkpoint_run = None
        else:
            checkpoint_file = self.config.get("EXPERIMENT", "checkpoint_file", fallback="")
            checkpoint_run = self.config.get("EXPERIMENT", "checkpoint_run", fallback="")

        if checkpoint_file:
            if not self.use_wandb:
                logging.warning("Could not load in checkpoint file, use_wandb is set to False")
            else:
                logging.info(f"Loading in checkpoint file: {checkpoint_file}")
                wandb_checkpoint = wandb.restore(checkpoint_file, run_path=checkpoint_run)
                checkpoint = torch.load(wandb_checkpoint.name)
                self.model.load_state_dict(checkpoint['learner_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                os.rename(os.path.join(wandb.run.dir, checkpoint_file),
                          os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))
        else:
            logging.info("No checkpoint used - learning from scratch")

        if self.use_wandb:
            # setting up metrics for logging to wandb
            # counter tracks number of batches of tasks seen by metalearner
            wandb.define_metric("num_epochs")

    def load_corpus(self):
        """ Load corpus being used """
        data_dir = self.config.get('DATASET', 'root_path', fallback="")
        if not data_dir:
            logging.exception('No training data specified.')
            raise Exception('No training data specified')

        fn = 'corpus.{}.data'.format(hashlib.md5(data_dir.encode()).hexdigest())
        if os.path.exists(fn):
            logging.info('Loading cached dataset...')
            corpus = torch.load(fn)
        else:
            logging.info('Producing dataset...')
            if "text8" in data_dir:
                corpus = data.Corpus(data_dir, self.sequence_length, truncate_long_utterances=False, raw_text=True)
            else:
                corpus = data.Corpus(data_dir, self.sequence_length, truncate_long_utterances=False, raw_text=False)
            torch.save(corpus, fn)
        return corpus

    def load_model(self):
        """ Either build model or load model from a checkpoint """
        # TODO: add model loading from checkpoint
        logging.info('Building model...')
        vocab_size = len(self.corpus.dictionary)
        hidden_size = self.config.getint('MODEL', 'hidden_size', fallback=64)
        n_layers = self.config.getint('MODEL', 'n_layers', fallback=16)
        dropout = self.config.getfloat('MODEL', 'dropout', fallback=0.1)
        tied = self.config.getboolean('MODEL', 'tied', fallback=False)
        inner_linear = self.config.getint('MODEL', 'inner_linear', fallback=2048)
        sequence_length = self.config.getint('TRAINING', 'sequence_length')
        
        model = next_char_transformer(vocab_size, hidden_size=hidden_size, n_layers=n_layers,
                                    dropout=dropout, tied=tied, max_sequence_len=sequence_length,
                                    intermediate_losses=True, inner_linear=inner_linear).to(self.base_device)

        num_params = sum([p.numel() for p in model.parameters()])
        logging.info('Number of parameters: {}'.format(num_params))

        return model

    def load_optimizer(self):
        """ Either build optimiser or load from a checkpoint """
        # TODO: add loading optimiser from checkpoint
        lr = self.config.getfloat('TRAINING', 'lr', fallback=0.003)
        momentum = self.config.getfloat('TRAINING', 'momentum', fallback=0.99)
        optimizer = optim.SGD(self.model.parameters(), lr, momentum)
        return optimizer

    def save_checkpoint(self, model_name):
        if not self.use_wandb:
            logging.error("Cannot save model checkpoint because use_wandb set to False")
        else:
            model_path = os.path.join(wandb.run.dir, model_name)
            logging.info(f"Saving model checkpoint to {model_path}")
            checkpoint = {
                'learner_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            # save latest checkpoint
            torch.save(checkpoint, model_path)

    def timeout_handler(self, signum, frame):
        """
        Gracefully handles early termination signals. Catches termination signals sent from  
        slurm just before the program is about to terminate and saves out a model checkpoint.
        """

        logging.info("Timeout (SIGINT) termination signal received")
        logging.info("Saving training state")

        if not self.use_wandb:
            logging.error("Cannot save epoch number because use_wandb set to False")
        else:
            checkpoint_file = os.path.join(wandb.run.dir, "latest-checkpoint.pt")
            if not os.path.exists(checkpoint_file):
                logging.error(f"No checkpoint file found at {checkpoint_file}, can't save state")
            else:
                logging.info(f"Saving latest checkpoint to wandb")
                wandb.save('latest-checkpoint.pt', policy="now")

                if not os.path.exists('tmp'):
                    os.mkdir('tmp')
                with open(f"tmp/{wandb.run.id}.runfile", "w+") as f:
                    logging.info(f"Saving current epoch number to tmp/{wandb.run.id}.runfile")
                    f.write(str(max(self.num_epochs, 0)))

        logging.info("Calling exit code 124 to trigger a rerun")
        exit(124)

    def __call__(self) -> None: 
        """ 
        Train or evaluate the model
        """

        ###############################################################################
        # Prepare experiment
        ###############################################################################

        log_interval = self.config.getint('LOGGING', 'log_interval', fallback=200)

        # metric for logging training data
        if self.use_wandb:
            wandb.define_metric("train.loss", step_metric="num_epochs", summary='min')
            wandb.define_metric("valid.loss", step_metric="num_epochs", summary='min')

        ###############################################################################
        # Batch data
        ###############################################################################

        batch_size = self.config.getint('TRAINING', 'batch_size')
        logging.info(f'Using batch_size = {batch_size}')
        eval_batch_size = batch_size
        test_batch_size = 1
        train_data = data.BatchedData(self.corpus.train, batch_size, self.sequence_length, self.base_device, is_train=True)
        val_data = data.BatchedData(self.corpus.valid, eval_batch_size, self.sequence_length, self.base_device, is_train=False)
        test_data = data.BatchedData(self.corpus.test, test_batch_size, self.sequence_length, self.base_device, is_train=False)

        ###############################################################################
        # Training code
        ###############################################################################

        best_val_loss = None
        epochs = self.config.getint('TRAINING', 'epochs', fallback=2)

        def evaluate(data_source, step=1):
            # Turn on evaluation mode which disables dropout.
            total_loss = AverageMeter()
            self.model.eval()
            with torch.no_grad():
                # Slide a window along of size `step`. Set step=self.sequence_length to evaluate per-utterance
                for batch, i in enumerate(range(0, data_source.data.size(0) - 1 - self.sequence_length, step)):
                    data, target, data_mask, target_mask = data_source.get_batch(i)
                    output = self.model(data, target_mask)
                    _, final_layer_loss = self.model.criterion(output, target)
                    total_loss.update(final_layer_loss.item(), data.size(0))
            return total_loss.avg

        def train():
            # Turn on training mode which enables dropout.
            self.model.train()
            current_loss = AverageMeter()
            total_loss = AverageMeter()
            start_time = time.time()
            for batch, i in enumerate(range(0, train_data.data.size(0) - 1, self.sequence_length)):
                data, target, data_mask, target_mask = train_data.get_batch(i)
                self.model.zero_grad()
                output = self.model(data, target_mask)
                average_loss_of_all_layers, final_layer_loss = self.model.criterion(output, target)
                average_loss_of_all_layers.backward()
                self.optimizer.step()

                current_loss.update(final_layer_loss.item(), data.size(0))
                total_loss.update(final_layer_loss.item(), data.size(0))

                if batch % log_interval == 0 and batch > 0:
                    avg_loss = current_loss.avg
                    elapsed = time.time() - start_time
                    logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        epoch, batch, len(train_data.data) // self.sequence_length,
                        elapsed * 1000 / log_interval, avg_loss,
                        math.exp(avg_loss), avg_loss / math.log(2)))
                    current_loss.reset()
                    start_time = time.time()

            return total_loss.avg

        ###############################################################################
        # Training loop
        ###############################################################################

        for epoch in range(self.num_epochs, epochs + 1):
            epoch_start_time = time.time()
            self.num_epochs = epoch
            train_loss = train()

            # Logging training results
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                'train ppl {:8.2f} | train bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                            train_loss, math.exp(train_loss), train_loss / math.log(2)))

            val_loss = evaluate(val_data, step=self.sequence_length//4)

            # Logging validation results
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss), val_loss / math.log(2)))
            logging.info('-' * 89)
            wandb.log({"train.loss": train_loss, "valid.loss": val_loss, "num_epochs": epoch},)
            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.config.getboolean('EXPERIMENT', 'save_best_model', fallback=True):
                    logging.info("New best model found - saving checkpoint")
                    self.save_checkpoint("best.pt")
                else:
                    logging.info("Didn't save best model seen so far - save_best_model set to False")

            # Let the model know how far through training we are for intermediate layer losses
            self.model.update(epoch // epochs)

            # Save a checkpoint
            if self.config.getboolean('EXPERIMENT', 'save_latest_checkpoint', fallback=True):
                logging.info("Saving a checkpoint at the end of the epoch")
                self.save_checkpoint("latest-checkpoint.pt")
            else:
                logging.error("Failed to save checkpoint - save_latest_checkpoint set to False")


        ###############################################################################
        # Final cleanup
        ###############################################################################

        logging.info("Finished training model")

        # Load the best saved model.
        if not self.use_wandb:
            logging.warning("Could not load in best model found, use_wandb is set to False")
        else:
            best_file = os.path.join(wandb.run.dir, f"best.pt")
            logging.info(f"Loading in best model seen so far from {best_file}")
            checkpoint = torch.load(best_file)
            self.model.load_state_dict(checkpoint['learner_state_dict'], strict=False)

        # Run on test data.
        test_loss = evaluate(test_data)
        logging.info('=' * 89)
        logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        logging.info('=' * 89)

        if self.config.getboolean('EXPERIMENT', 'save_final_model', fallback=True):
            logging.info("Saving final model")
            self.save_checkpoint("final.pt")
        else:
            logging.info("Could not save final model - save_final_model set to False")
