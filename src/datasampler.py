""" Module for custom data samplers. """

import copy
from typing import Callable, Iterator, Sequence, Union

# typing imports
from torch.utils.data import BatchSampler

class CustomBatchSampler(BatchSampler):
    """
    Custom batch sampler that ensures we get enough data to fill a batch once the collator has joined utterances together to sequence of length max_seq_length.
    """

    def __init__(self, sampler, batch_size, drop_last, max_seq_length):
        """
        Args:
            sampler: (Sampler): The sampler to use.
            batch_size: (int): The batch size.
            drop_last: (bool): Whether to drop the last batch.
            max_seq_len: (int): The maximum sequence length.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.max_seq_len = max_seq_length
        self.total_batch_size = batch_size * max_seq_length

    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                batch = []
                total_len = 0
                try:
                    while total_len < self.total_batch_size:
                        idx = next(sampler_iter)
                        batch.append(idx)
                        length = len(self.sampler.data_source[idx]["input_ids"])
                        total_len += length
                    yield batch[:-1]
                    batch = [idx]
                    total_len = length
                except StopIteration:
                    break
        else:
            batch = []
            idx_in_batch = 0
            total_len = 0
            for idx in self.sampler:
                batch.append(idx)
                length = len(self.sampler.data_source[idx]["input_ids"])
                total_len += length
                idx_in_batch += 1
                if total_len >= self.total_batch_size:
                    yield batch[:-1]
                    idx_in_batch = 1
                    total_len = length
                    batch = [idx]
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
