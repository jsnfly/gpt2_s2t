import torch
import torch.nn as nn

class S2TModel(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.hidden_dim = self.decoder.config.n_embd
        self.projection_layer = nn.Linear(1024, self.hidden_dim)
        self.projection_layer_ln = nn.LayerNorm(self.hidden_dim, eps=self.decoder.config.layer_norm_epsilon)

        # Use same initialization as in GPT-2.
        self.projection_layer.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)

    def forward(self, encoder_hidden_states, input_ids):
        encoder_states = self.projection_layer(encoder_hidden_states)

        # Add position encodings. (Seems to help a lot to use the already trained decoder embeddings.)
        encoder_states += self.decoder.transformer.wpe(torch.arange(0, encoder_hidden_states.shape[1],
                                                                    device=input_ids.device))
        encoder_states = self.projection_layer_ln(encoder_states)
        return self.decoder(input_ids, encoder_hidden_states=encoder_states, labels=input_ids)
