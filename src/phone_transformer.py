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


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

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
    
    def __init__(self, resubmit_after=None):
        """ Initialize base model based using wandb config """

        self.setup_seed()
        self.resubmit_after = resubmit_after
        if resubmit_after:
            self.start_time = time.time()
            logging.info(f'Job set to stop and resubmit after {resubmit_after} hours')
        else:
            logging.info(f'Job not set to resubmit, use --resubmit_after if you need automatic resubmission')

        self.base_device = wandb.config.get('device', DEFAULT_DEVICE)
        logging.info(f'Running phone transformer on device: {self.base_device}')

        # Using the number of log steps from wandb as number of epochs. If not resuming, will be 0
        self.num_epochs = wandb.run.step

        # Load corpus, model and optimiser
        self.sequence_length = wandb.config.get('sequence_length')
        self.corpus = self.load_corpus()
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()

        # if num_epochs is 0 at the start of training, then we are resuming training 
        if self.num_epochs > 0:
            checkpoint_file = "latest-checkpoint.pt"
        else:
            checkpoint_file = wandb.config.get('checkpoint_file', '')

        if checkpoint_file:
            logging.info(f'Loading in checkpoint file: {checkpoint_file}')
            checkpoint = torch.load(wandb.restore(checkpoint_file).name)
            self.model.load_state_dict(checkpoint['learner_state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            os.rename(os.path.join(wandb.run.dir, checkpoint_file),
                os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))
        else:
            logging.info('No checkpoint used - learning from scratch')
            self.num_epochs = 0

    def setup_seed(self):
        """ Set up the seed """
        seed = wandb.config.get('seed',-1)
        if seed < 0: 
            logging.info('Skipping seed setting for reproducibility')
            logging.info('If you would like to set a seed, set seed to a positive value in config')
            return
        logging.info(f'Setting seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() > 0:
            torch.cuda.manual_seed_all(seed)

    def load_corpus(self):
        """ Load corpus being used """
        data_dir = wandb.config.get('root_path', '')
        if not data_dir:
            logging.exception('No training data specified.')
            raise Exception('No training data specified')

        fn = 'corpus.{}.data'.format(hashlib.md5(data_dir.encode()).hexdigest())
        if os.path.exists(fn):
            logging.info(f'Loading cached dataset at {fn}')
            corpus = torch.load(fn)
        else:
            logging.info('Producing dataset...')
            if 'text8' in data_dir:
                corpus = data.Corpus(data_dir, self.sequence_length, truncate_long_utterances=False, raw_text=True)
            else:
                corpus = data.Corpus(data_dir, self.sequence_length, truncate_long_utterances=False, raw_text=False)
            torch.save(corpus, fn)
        return corpus

    def load_model(self):
        # TODO: Maybe just pass in the MODEL dict directly?
        logging.info('Building model...')
        vocab_size = len(self.corpus.dictionary)
        hidden_size = wandb.config.get('hidden_size', 64)
        n_layers = wandb.config.get('n_layers', 16)
        dropout = wandb.config.get('dropout', 0.1)
        inner_linear = wandb.config.get('inner_linear', 2048)
        sequence_length = wandb.config.get('sequence_length')
        
        model = next_char_transformer(vocab_size, hidden_size=hidden_size, n_layers=n_layers,
                                    dropout=dropout, max_sequence_len=sequence_length,
                                    intermediate_losses=True, inner_linear=inner_linear).to(self.base_device)

        num_params = sum([p.numel() for p in model.parameters()])
        logging.info('Number of parameters: {}'.format(num_params))

        return model

    def load_optimizer(self):
        lr = wandb.config.get('lr', 0.003)
        momentum = wandb.config.get('momentum', 0.99)
        optimizer = optim.SGD(self.model.parameters(), lr, momentum)
        return optimizer

    def save_checkpoint(self, model_name):
        """ Save a checkpoint and immediately upload it to wandb in case we get terminated """
        model_path = os.path.join(wandb.run.dir, model_name)
        logging.info(f'Saving model checkpoint to {model_path}')
        checkpoint = {
            'learner_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)
        logging.info(f'Saving latest checkpoint to wandb')
        # wandb.save(model_path, policy='now')

    def resubmit_job(self):
        """
        Submit a new slurm job to continue training. Bad practise to save a command to a file to be run later but works for now.
        """

        saved_config = os.path.join(wandb.run.dir, 'config.yaml')
        run_id = wandb.run.id
        command = f'sbatch scripts/slurm_submit.wilkes3 {saved_config} {run_id}'
        logging.info('Ending current wandb run')
        wandb.finish()
        logging.info('Submitting a new job to resume training')
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        while os.path.exists('tmp/requeue.sh'):
            time.sleep(10)
            logging.info('tmp/requeue.sh already exists - waiting for another process to finish using it')
        logging.info(f'Saving command to tmp/requeue.sh: "{command}"')
        with open('tmp/requeue.sh', 'w') as f:
            f.writelines(command)
        logging.info('Exiting with error code 124 to requeue job')
        exit(124)

    def log_losses(self, loss, message='', last_time=None):
        logging.info('=' * 89)
        if last_time:
            message = message + ' | time: {:5.2f}s'.format(time.time() - last_time)
        logging.info('| {} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
            message, loss, math.exp(loss), loss / math.log(2)))
        logging.info('=' * 89)

    def __call__(self) -> None: 
        """ 
        Train or evaluate the model
        """

        ###############################################################################
        # Prepare experiment
        ###############################################################################

        log_interval = wandb.config.get('log_interval', 200)

        # metric for logging training data
        # wandb.define_metric('epoch')
        # wandb.define_metric("train.loss", step_metric='epoch', summary='min')
        # wandb.define_metric("valid.loss", step_metric='epoch', summary='min')

        ###############################################################################
        # Batch data
        ###############################################################################

        batch_size = wandb.config.get('batch_size')
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
        epochs = wandb.config.get('epochs', 2)

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
            train_loss = train()
            self.log_losses(train_loss, 'end of epoch {:3d} | TRAIN STATS'.format(epoch), epoch_start_time)

            val_loss = evaluate(val_data, step=self.sequence_length)
            self.log_losses(val_loss, 'end of epoch {:3d} | VALID STATS'.format(epoch), epoch_start_time)

            # Log to WandB. This also increases wandb.run.step by 1,
            # which is used for setting resumed epoch number
            wandb.log({"train" : {"loss": train_loss}, 'valid': {'loss': val_loss}})
            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                if wandb.config.get('save_best_model', True):
                    logging.info("New best model found - saving checkpoint")
                    self.save_checkpoint("best.pt")
                else:
                    logging.info("Didn't save best model seen so far - save_best_model set to False")

            # Let the model know how far through training we are for intermediate layer losses
            self.model.update(epoch // epochs)

            # Save a checkpoint
            if wandb.config.get('save_latest_checkpoint', True):
                logging.info("Saving a checkpoint at the end of the epoch")
                self.save_checkpoint("latest-checkpoint.pt")
            else:
                logging.error("Failed to save checkpoint - save_latest_checkpoint set to False")

            # Check if it's time to resubmit
            if self.resubmit_after:
                if time.time() - self.start_time > self.resubmit_after * 3600:
                    logging.info('Ellapsed maximum time for job, resubmitting.')
                    self.resubmit_job()


        ###############################################################################
        # Final cleanup
        ###############################################################################

        logging.info("Finished training model")

        # Load the best saved model.
        best_file = os.path.join(wandb.run.dir, f"best.pt")
        logging.info(f"Loading in best model seen so far from {best_file}")
        checkpoint = torch.load(best_file)
        self.model.load_state_dict(checkpoint['learner_state_dict'], strict=False)

        # Run on test data.
        test_loss = evaluate(test_data, step=self.sequence_length)

        # Log final loss
        self.log_losses(test_loss, 'End of training | TEST STATS ')
        wandb.log({'test_loss': test_loss})

        if wandb.config.get('save_final_model', True):
            logging.info("Saving final model")
            self.save_checkpoint("final.pt")
        else:
            logging.info("Could not save final model - save_final_model set to False")
