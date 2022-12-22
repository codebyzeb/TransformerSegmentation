import collections
import copy
import torch
import torch.nn as nn

from .annotated_attention import AddPositionalEncoding, MultiHeadedAttention, SublayerConnection, clones, PositionwiseFeedForward, Generator, Embeddings

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout,
                 intermediate_layer_predictions=True, generator=None, max_sequence_len=512, force_prediction=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.add_positional_encoding = AddPositionalEncoding(size, max_sequence_len)
        self.norm = self.sublayer[0].norm

        self.size = size
        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.force_prediction = force_prediction
        if intermediate_layer_predictions and self.training:
            self.classifier = copy.deepcopy(generator)

    def forward(self, x, mask):
        x = self.add_positional_encoding(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        if self.force_prediction or (self.intermediate_layer_predictions and self.training):
            return x, self.classifier(self.norm(x))
        else:
            return x, None


class Encoder(nn.Module):
    """ Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers, intermediate_layer_predictions=True):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        # enforce a prediction for the last layer
        self.layers[-1].force_prediction = True
        self.norm = nn.LayerNorm(layer.size)
        self.intermediate_layer_predictions = intermediate_layer_predictions

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        intermediate_predictions = []
        for layer in self.layers:
            x, prediction = layer(x, mask)
            intermediate_predictions.append(prediction)
        return self.norm(x), intermediate_predictions


class MultiLayerCrossEntropy(nn.Module):
    """ Cross entropy loss for multiple layers. """
    def __init__(self, vocab_size, *args, **kwargs):
        super(MultiLayerCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, layer_outputs, target):
        total_loss = torch.zeros(1, dtype=layer_outputs[-1].dtype, device=layer_outputs[-1].device)
        n_layers_with_loss = 0
        all_losses = []
        for layer_output in layer_outputs:
            if layer_output is not None:
                # For both evaluation and training, we take the loss across the whole predicted sequence
                loss = self.cross_entropy(layer_output.view(-1, self.vocab_size).contiguous(), target)
                total_loss += loss
                n_layers_with_loss += 1
                all_losses.append(loss)

        average_loss_of_all_layers = total_loss / n_layers_with_loss
        final_layer_loss = loss
        return final_layer_loss, average_loss_of_all_layers, all_losses


class NextCharTransformer(nn.Module):
    """ Next character transformer model. """

    def __init__(self, vocab_size, n_layers=64,
                 hidden_size=512, inner_linear=2048,
                 n_heads=8, dropout=0.55, max_sequence_len=512, ignore_index=-100,
                 intermediate_layer_predictions=True):
        super(NextCharTransformer, self).__init__()

        attn = MultiHeadedAttention(n_heads, hidden_size, dropout)
        ff = PositionwiseFeedForward(hidden_size, inner_linear, dropout)

        generator = Generator(hidden_size, vocab_size)
        self.encoder = Encoder(EncoderLayer(hidden_size, copy.deepcopy(attn), copy.deepcopy(ff),
                                            dropout, intermediate_layer_predictions, generator,
                                            max_sequence_len),
                               n_layers, intermediate_layer_predictions)
        self.embed = Embeddings(hidden_size, vocab_size)

        # Set ignore_index to avoid training on padding token
        self.criterion = MultiLayerCrossEntropy(vocab_size, ignore_index=ignore_index)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.n_layers = n_layers
        self.num_intermediate_losses = n_layers if intermediate_layer_predictions else 1

    def forward(self, src, mask):
        """ Take in and process masked src and target sequences. """
        src_emb = self.embed(src)
        emb, intermediate_predictions = self.encoder(src_emb, mask)
        return intermediate_predictions

    def update(self, training_percent):
        """ Stop using losses from intermediate layer as function of time in training.
        See section 2.1 - Intermediate Layer Losses from Character Transformer paper. 
        """

        self.num_intermediate_losses = 1
        for i, layer in enumerate(self.encoder.layers[:-1]):
            if training_percent > ((i+1) / (2 * self.n_layers)):
                layer.intermediate_layer_predictions = False
            else:
                self.num_intermediate_losses += 1


def next_char_transformer(src_vocab, n_layers=64, hidden_size=512,
                          inner_linear=2048, n_heads=8, dropout=0.55, ignore_index=-100,
                          max_sequence_len=512, intermediate_losses=True):
    """ Construct a next character transformer model. """
    return NextCharTransformer(src_vocab, n_layers, hidden_size,
                               inner_linear, n_heads, dropout, max_sequence_len,
                               ignore_index, intermediate_losses)
