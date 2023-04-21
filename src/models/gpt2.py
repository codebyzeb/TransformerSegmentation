import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .registry import register_model

""" 
Model classes use the GPT-2 Architecture 
"""

from transformers import (
    GPT2Config,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
)

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


@register_model("probe_gpt2_lm_head_model", GPT2Config)
class GPT2Probe(GPT2ForTokenClassification):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        We override the forward pass, freezing the transformer layer but otherwise using the same code as the original
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # Freeze the transformer layer
        with torch.no_grad():
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
