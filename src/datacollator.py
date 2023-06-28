from transformers.data import DataCollatorForLanguageModeling
from torch.utils.data import BatchSampler

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Custom Data Collator that randomly joins utterances together to form longer sequences.
    """

    def __init__(self, tokenizer, max_seq_length=512, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.max_seq_length = max_seq_length

    def __call__(self, examples, return_tensors=None):
        """
        Args:
            examples: (List[Dict[str, List[int]]]): The examples to collate.
        Returns:
            (Dict[str, torch.Tensor]): The collated examples.
        """
        new_examples = []
        keys = list(examples[0].keys())
        long_examples = {}
        for key in keys:
            long_examples[key] = []
        for example in examples:
            for key in keys:
                long_examples[key].extend(example[key])
        for i in range(0, len(long_examples[keys[0]]), self.max_seq_length):
            new_example = {}
            for key in keys:
                new_example[key] = long_examples[key][i : i + self.max_seq_length]
            new_examples.append(new_example)
        return super().__call__(new_examples, return_tensors=return_tensors)
