from .registry import register_model

""" 
Model classes use the GPT-2 Architecture 
"""

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

### Wrapping the GPT2 models to make them compatible with the model registry ###


@register_model("gpt2_lm_head_model", GPT2Config)
class GPT2LMHeadModel(GPT2LMHeadModel):
    pass


### Custom GPT2 Models ###

# NOTE: The forward pass of these models always needs to return ModelOutput
#       objects. See the documentation for more details.


@register_model("tuned_gpt2_lm_head_model", GPT2Config)
class TunedGPT2LMHeadModel(GPT2LMHeadModel):
    pass
    # TODO: Insert any code to overwrite the standard behavior
