__author__ = 'Zeb Goriely'
""" Wrapper class for training and evaluating a phoneme transformer model """

import logging 
import time 
import os
import wandb

import math
import random

import numpy as np
import torch
import torch.optim as optim
import torch.onnx

from .data import data
from .model.model import next_char_transformer
from .segmentation.segment import Segmenter


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """ Computes and stores the average and current value """
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
    """ Orchestrates model loading, training and evaluation using the next char transformer. """
    
    def __init__(self, resubmit_after=None):
        """ Initialize base model based using wandb config
        Parameters
        ----------
        resubmit_after : int
            Number of hours to run before resubmitting job
        """

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
        self.log_interval = wandb.config.get('log_interval', 200)
        self.clip = wandb.config.get('clip', 1.0)
        self.corpus = self.load_corpus()
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()
        self.try_resume_training()

    def try_resume_training(self):
        """ Try to resume training if a checkpoint exists """
        if self.num_epochs > 0:
            logging.info(f'Resuming training from epoch {self.num_epochs}')
            checkpoint_file = "latest-checkpoint.pt"
        else:
            checkpoint_file = wandb.config.get('checkpoint_file', '')
            logging.info(f'Found checkpoint file found in config: {checkpoint_file}')

        if checkpoint_file:
            logging.info(f'Loading in checkpoint file: {checkpoint_file}')
            checkpoint = torch.load(wandb.restore(checkpoint_file).name)
            self.model.load_state_dict(checkpoint['learner_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            os.rename(os.path.join(wandb.run.dir, checkpoint_file), os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))  
        else:
            logging.info('No checkpoint found - training from scratch')

    def setup_seed(self):
        """ Set up the seed """
        seed = wandb.config.get('seed', -1)
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
            logging.exception('No training data specified')
            raise Exception('No training data specified')

        fn = 'corpus.{}.data'.format('.'.join(data_dir.split('/')))
        if os.path.exists(fn):
            logging.info(f'Loading cached dataset at {fn}')
            corpus = torch.load(fn)
            return corpus
        else:
            logging.info('Producing dataset...')
            if 'text8' in data_dir:
                logging.info('Using Raw Tokenizer for text8')
                tokenizer = data.RawTokenizer()
            else:
                logging.info('Using Space Tokenizer and removing word boundary token')
                tokenizer = data.SpaceTokenizer(banned_tokens=[';eword'])
            corpus = data.Corpus(data_dir, tokenizer)
            torch.save(corpus, fn)
        return corpus

    def load_model(self):
        """ Load model being used """
        logging.info('Building model...')
        vocab_size = len(self.corpus.dictionary)
        hidden_size = wandb.config.get('hidden_size', 64)
        n_layers = wandb.config.get('n_layers', 16)
        dropout = wandb.config.get('dropout', 0.1)
        inner_linear = wandb.config.get('inner_linear', 2048)
        sequence_length = wandb.config.get('sequence_length')

        # Don't train on <PAD>
        ignore_index = 0
        
        model = next_char_transformer(vocab_size, hidden_size=hidden_size, n_layers=n_layers,
                                    dropout=dropout, max_sequence_len=sequence_length, ignore_index=ignore_index,
                                    intermediate_losses=True, inner_linear=inner_linear).to(self.base_device)

        num_params = sum([p.numel() for p in model.parameters()])
        logging.info('Number of parameters: {}'.format(num_params))

        return model

    def load_optimizer(self):
        """ Load optimizer being used """
        lr = wandb.config.get('lr', 0.003)
        momentum = wandb.config.get('momentum', 0.99)
        optimizer = optim.SGD(self.model.parameters(), lr, momentum)
        return optimizer

    def save_checkpoint(self, model_name):
        """ Save a checkpoint and upload it to wandb in case we get terminated """
        self.model.zero_grad()
        self.optimizer.zero_grad()
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
        """ Submit a new slurm job to continue training. Bad practise to save a command to a file to be run later but works for now. """
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
        """ Log the losses """
        logging.info('=' * 89)
        if last_time:
            message = message + ' | time: {:5.2f}s'.format(time.time() - last_time)
        logging.info('| {} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
            message, loss, math.exp(loss), loss / math.log(2)))
        logging.info('=' * 89)

    def step(self, data_source, train=False, step=1):
        """ Take a step in the model """
        self.model.train() if train else self.model.eval()

        # Logging information
        current_loss = AverageMeter()
        total_loss = AverageMeter()
        layer_losses = [AverageMeter() for i in range(self.model.n_layers)]
        start_time = time.time()

        for batch, i in enumerate(range(0, data_source.data.size(0) - 1 - self.sequence_length, step)):

            if train:
                # For training, we sample batches randomly from the data
                # TODO: CHECK IF KEEPING UTTERANCES SEPARATE
                #if self.corpus.keep_utterances_separate:
                #    i = torch.randint(low=0, high=(train_data.data.size(0) // self.sequence_length), size=(1,)).long().item() * self.sequence_length
                # else:
                i = torch.randint(low=0, high=(data_source.data.size(0) - self.sequence_length), size=(1,)).long().item()
                self.optimizer.zero_grad()

             # Get batch, run through model, get loss and step optimizer
            data, target, _, target_mask = data_source.get_batch(i)
            output = self.model(data, target_mask)
            final_layer_loss, average_loss_of_all_layers, all_layer_losses = self.model.criterion(output, target)
            if train:
                average_loss_of_all_layers.backward() # Train model on all layer losses, not just final layer
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
            
            # Logging information
            current_loss.update(final_layer_loss.item(), data.size(0))
            total_loss.update(final_layer_loss.item(), data.size(0))
            for i in range(len(all_layer_losses)):
                layer_losses[-1-i].update(all_layer_losses[-1-i].item(), data.size(0))

            # Logging
            if train and batch % self.log_interval == 0 and batch > 0:
                avg_loss = current_loss.avg
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    wandb.run.step, batch, len(data_source.data) / step,
                    elapsed * 1000 / self.log_interval, avg_loss,
                    math.exp(avg_loss), avg_loss / math.log(2)))
                current_loss.reset()
                start_time = time.time()

        return total_loss.avg, [layer_loss.avg for layer_loss in layer_losses]
  
    def __call__(self) -> None: 
        """ Train or evaluate the model """

        ###############################################################################
        # Prepare experiment
        ###############################################################################

        segment_interval = wandb.config.get('segment_interval', 50)

        ###############################################################################
        # Batch data
        ###############################################################################

        batch_size = wandb.config.get('batch_size')
        logging.info(f'Using batch_size = {batch_size}')
        eval_batch_size = batch_size
        test_batch_size = batch_size
        train_data = data.BatchedData(self.corpus.train, batch_size, self.sequence_length, self.base_device)
        val_data = data.BatchedData(self.corpus.valid, eval_batch_size, self.sequence_length, self.base_device)
        test_data = data.BatchedData(self.corpus.test, test_batch_size, self.sequence_length, self.base_device)

        ###############################################################################
        # Training code
        ###############################################################################

        best_val_loss = None
        epochs = wandb.config.get('epochs', 2)

        for epoch in range(self.num_epochs, epochs + 1):

            # Let the model know how far through training we are for intermediate layer losses
            self.model.update(epoch / epochs)
            logging.info(f'Training on losses from final {self.model.num_intermediate_losses} layers')

            epoch_start_time = time.time()
            train_loss, layer_losses = self.step(train_data, train=True, step=self.sequence_length)
            layer_losses = {f"layer_{i}" : layer_losses[i] for i in range(len(layer_losses))}
            self.log_losses(train_loss, 'end of epoch {:3d} | TRAIN STATS'.format(epoch), epoch_start_time)

            val_loss, _ = self.step(val_data, train=False, step=self.sequence_length)
            self.log_losses(val_loss, 'end of epoch {:3d} | VALID STATS'.format(epoch), epoch_start_time)

            # Every `segment_interval` epochs, we also check segmentation performance
            # TODO: ADD HERE

            # Log to WandB. This also increases wandb.run.step by 1,
            # which is used for setting resumed epoch number
            log_dict = {'train' : {"loss": train_loss},
                        'valid': {'loss': val_loss},
                        'layer_losses' : layer_losses,
                        'num_intermediate_losses' : self.model.num_intermediate_losses,
                        'epoch' : epoch}
            
            wandb.log(log_dict)
            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                if wandb.config.get('save_best_model', True):
                    logging.info("New best model found - saving checkpoint")
                    self.save_checkpoint("best.pt")
                else:
                    logging.info("Didn't save best model seen so far - save_best_model set to False")

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

        logging.info("Finished training model")

        ###############################################################################
        # Run on test data
        ###############################################################################

        best_file = os.path.join(wandb.run.dir, f"best.pt")
        logging.info(f"Loading in best model seen so far from {best_file}")
        checkpoint = torch.load(best_file)
        self.model.load_state_dict(checkpoint['learner_state_dict'], strict=False)
        test_loss, _ = self.step(test_data, train=False, step=1)
        self.log_losses(test_loss, 'End of training | TEST STATS ')
        wandb.log({'test_loss': test_loss})
        if wandb.config.get('save_final_model', True):
            logging.info("Saving final model")
            self.save_checkpoint("final.pt")
        else:
            logging.info("Could not save final model - save_final_model set to False")

