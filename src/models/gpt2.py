import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, LayerNorm, ModuleList, Sequential, Module
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn.parameter import Parameter

from .registry import register_model

""" 
Model classes use the GPT-2 Architecture 
"""

from transformers import (
    GPT2Config,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
    GPT2PreTrainedModel,
    GPT2Model,
)

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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


# Custom Linear layer that stores the weight matrix transposed,
# for the purpose of tieing with the output linear layer
class CustomLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.T, self.bias)

class FeatureMap(Module):
    def __init__(self, feature_map, device=None, dtype=torch.float32):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.tensor([feature_map[i] for i in range(len(feature_map))], **factory_kwargs), requires_grad=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight)

@register_model("gpt2_feature_model", GPT2Config)
class GPT2FeatureModel(GPT2PreTrainedModel):
    """
    Implementation of GPT2LMHeadModel with feature vectors as I/O instead of vocab indices
    """

    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
        r"custom_embedding.weight",
        r"custom_embedding.bias",
        r"lm_head.bias",
        r"embedding_norm.weight",
        r"embedding_norm.bias",
    ]

    def __init__(self, config, feature_map=None):
        """
        Args:
            config: GPT2Config object
            feature_map: mapping from vocab index to feature vector
        """
        super().__init__(config)

        self.vocab_size = config.vocab_size

        # Used to flip the output logits of forward() back to vocab indices for the purpose of generation
        self.return_token_logits = False

        # Feature map should be provided when initialising, but when reloading from checkpoint
        # we need to set it to None and it'll be initialised from the checkpoint
        if feature_map is None:
            feature_map = {i: [0]*39 for i in range(config.vocab_size)}
        
        # Feature Map layer - maps from token ids to feature vectors. Doesn't update during training.
        self.feature_map = FeatureMap(feature_map)
        self.feature_size = self.feature_map.weight.shape[1]

        # Embedding layer - map from feature vector to embedding with layer norm
        self.custom_embedding = Sequential(
            CustomLinear(self.feature_size, config.n_embd),
            LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        )

        # Transformer unchanged
        self.transformer = GPT2Model(config)

        # Instead of mapping to vocab, we have a prediction head for each feature
        self.lm_heads = ModuleList([Linear(config.n_embd, 3, bias=False) for _ in range(self.feature_size)])

        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ADDED BEHAVIOUR - Instead of passing input_ids to transformer, we map to feature vector
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.feature_map(input_ids)
            inputs_embeds = self.custom_embedding(inputs_embeds)
            # If less than 2 dimensions, unsqueeze
            if len(inputs_embeds.shape) < 3:
                inputs_embeds = inputs_embeds.unsqueeze(dim=0)
            input_ids = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_heads[0].weight.device)

        # ADDED BEHAVIOUR - multiple lm head predict features, instead a single lm head to predict a token
        feature_logits = [lm_head(hidden_states) for lm_head in self.lm_heads]

        loss = None
        if labels is not None:
            loss = 0
            label_vectors = self.feature_map(labels)
            for i, logits in enumerate(feature_logits):
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = label_vectors[..., 1:, i].contiguous().long()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / self.feature_size

        # Generate a tensor for token ids that has the same shape as logits
        if self.return_token_logits:
            shape = list(feature_logits[0].shape)
            shape[-1] = self.vocab_size
            log_probs = torch.zeros(shape, device=feature_logits[0].device)
            feature_matrix_long = self.feature_map.weight.long()
            for feature in range(self.feature_size):
                feature_probs = F.log_softmax(feature_logits[feature], dim=-1)
                log_probs += feature_probs[:, :, feature_matrix_long[:, feature]]
            log_probs += math.log(1 / self.feature_size)
            logits = log_probs - torch.log(1 - torch.exp(log_probs))  # Convert back to logits
        else:
            logits = torch.stack(feature_logits, dim=1)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
