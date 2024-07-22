from transformers import LlamaConfig, LlamaForCausalLM

from .registry import register_model

""" 
Model classes using the Llama Architecture 
"""

### Wrapping the Llama models to make them compatible with the model registry ###


@register_model("llama_lm", LlamaConfig)
class LlamaForCausalLM(LlamaForCausalLM):
    pass