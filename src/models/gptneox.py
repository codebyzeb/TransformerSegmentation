from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from .registry import register_model

""" 
Model classes using the GPTNeoX Architecture 
"""

### Wrapping the GPTNeoX models to make them compatible with the model registry ###


@register_model("gptneox_lm", GPTNeoXConfig)
class GPTNeoXForCausalLM(GPTNeoXForCausalLM):
    pass