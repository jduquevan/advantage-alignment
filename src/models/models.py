import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class GruModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1):
        super(GruModel, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size
        
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.to(device)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        output, x = self.gru(x.unsqueeze(1), h_0)
        return output, x
    
    def get_embeds(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return x

class LstmModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1):
        super(LstmModel, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size
        
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.to(device)

    def forward(self, x, h_0=None):
        self.lstm.flatten_parameters()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # Reshape x to match LSTM input shape
        x = x.unsqueeze(1)
        output, (h_n, c_n) = self.lstm(x, h_0)
        return output, (h_n, c_n)

# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
# Simple implementation of https://arxiv.org/pdf/1910.06764.pdf
class StableTransformer(nn.Module):
    def __init__(
        self,
        in_size, 
        d_model, 
        device,
        dim_feedforward=40,  
        num_layers=1, 
        nhead=4, 
        max_seq_len=50, 
        dropout=0.1
        ):

        super(StableTransformer, self).__init__()
        self.layers = nn.ModuleList()
        
        self.in_size = in_size
        self.d_model = d_model
        self.device = device

        self.num_layers = num_layers
        self.n_heads = nhead
        self.max_seq_len = max_seq_len

        self.embed_layer = nn.Linear(in_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        for _ in range(self.num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_first=True
            )
            # Move the layer to the specified device
            encoder_layer = encoder_layer.to(device)
            self.layers.append(encoder_layer)

    def forward(self, src, partial_forward=True):
        src = self.embed_layer(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = src
        for layer in self.layers:
            output = layer(output) + src
        if partial_forward:
            output = output[-1, :, :]
        return output

    def get_embeds(self, src):
        src = self.embed_layer(src)
        return src

class F1StableTransformer(nn.Module):
    def __init__(
        self,
        in_size, 
        d_model, 
        device,
        dim_feedforward=40,  
        num_layers=1,
        num_embed_layers=1,
        nhead=4, 
        max_seq_len=100, 
        dropout=0.1
        ):

        super(F1StableTransformer, self).__init__()
        self.layers = nn.ModuleList()
        
        self.in_size = in_size
        self.d_model = d_model
        self.device = device

        self.num_layers = num_layers
        self.n_heads = nhead
        self.max_seq_len = max_seq_len

        self.embed_state_layer = nn.Linear(in_size, d_model)
        self.embed_action_layer = nn.Linear(4, d_model)

        self.embed_layers = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(num_embed_layers)])

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        for _ in range(self.num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_first=True
            )
            # Move the layer to the specified device
            encoder_layer = encoder_layer.to(device)
            self.layers.append(encoder_layer)

    def forward(self, states, actions, partial_forward=True):
        B, T, _ = states.shape
        H = self.d_model
        embed_states = self.embed_state_layer(states)
        embed_actions = self.embed_action_layer(actions)

        for embed_layer in self.embed_layers:
            embed_states = F.relu(embed_layer(embed_states))
            embed_actions = F.relu(embed_layer(embed_actions))

        # Interleave states and actions: s, a, s, a, ...
        src = torch.stack((embed_states, embed_actions), dim=2).view(B, -1, H)

        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = src
        for layer in self.layers:
            output = layer(output) + src
        if partial_forward:
            output = output[:, -2, :]
        return output

    def get_embeds(self, states, actions):
        B, T, _ = states.shape
        H = self.d_model
        embed_states = self.embed_state_layer(states)
        embed_actions = self.embed_action_layer(actions)

        for embed_layer in self.embed_layers:
            embed_states = F.relu(embed_layer(embed_states))
            embed_actions = F.relu(embed_layer(embed_actions))

        src = torch.stack((embed_states, embed_actions), dim=2).view(B, -1, H)

        return src

class MLPModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, gru=None, transformer=None):
        super(MLPModel, self).__init__()
        self.gru = gru
        self.transformer = transformer

        self.use_gru = self.gru != None

        self.hidden = nn.Linear(in_size, hidden_size)

        # Termination head
        self.term_head = nn.Linear(hidden_size, 2)  # Assuming 2 categories for the term

        # Communication head
        self.utte_head_mean = nn.Linear(hidden_size, 6)
        self.utte_head_std = nn.Linear(hidden_size, 6)

        # Proposition head
        self.prop_head_mean = nn.Linear(hidden_size, 3)
        self.prop_head_std = nn.Linear(hidden_size, 3)

        self.to(device)

    def forward(self, x, h_0=None, partial_forward=True):
        output = None
        if self.use_gru:
            output, x = self.gru(x, h_0)
        else:
            x = self.transformer(x, partial_forward)
        x = F.relu(self.hidden(F.relu(x)))

        # Term
        term_probs = F.softmax(self.term_head(x), dim=-1)

        # Utte
        utte_mean = self.utte_head_mean(x)  
        utte_log_std = F.softplus(self.utte_head_std(x))

        # Prop
        prop_mean = self.prop_head_mean(x)
        prop_log_std = F.softplus(self.prop_head_std(x))

        return output, term_probs, (utte_mean, utte_log_std), (prop_mean, prop_log_std)

    
class MLPModelDiscrete(nn.Module):
    def __init__(self, in_size, device, output_size, num_layers=1, hidden_size=40, encoder=None, use_gru=True):
        super(MLPModelDiscrete, self).__init__()
        # self.transformer = transformer
        self.num_layers = num_layers
        self.encoder = encoder
        self.use_gru = use_gru

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])

        # Termination head
        # self.term_head = nn.Linear(hidden_size, 2)  # Assuming 2 categories for the term

        # Communication head
        # self.utte_head = nn.Linear(hidden_size, 6)

        # Proposition heads
        assert output_size == 6 , f"when designing the game we fixated on 5 being max items, if that changed, remove this: output_size now is: {output_size}"
        self.prop_heads = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(3)])
        self.temp = nn.Parameter(torch.ones(1))

        self.to(device)

    def forward(self, x, h_0=None, partial_forward=True):
        output = None
        # item_and_utility = x[:, :, 0:6]
        prop = x['prop']
        item_and_utility = x['item_and_utility']
        if self.use_gru:
            output, x = self.encoder(prop, h_0)
            x = x.squeeze(0)
        else:
            x = self.encoder(prop, partial_forward)

        x = x/torch.linalg.norm(x)

        if partial_forward:
            x = torch.cat([
                item_and_utility,
                x
            ], dim=-1)
        else:
            x = torch.cat([
                item_and_utility.permute((1, 0, 2)),
                x
            ], dim=-1)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Term
        # term_probs = F.softmax(self.term_head(x), dim=-1)
        term_probs = None

        # Utte
        # utte_probs = F.softmax(self.utte_head(x), dim=-1)
        utte_probs = None

        # Prop
        prop_probs = [F.softmax(prop_head(x)/ self.temp, dim=-1) for prop_head in self.prop_heads]

        return output, term_probs, utte_probs, prop_probs
    
class F1MLPModel(nn.Module):
    def __init__(self, in_size, device, num_layers=1, hidden_size=40, encoder=None, use_gru=True):
        super(F1MLPModel, self).__init__()

        self.num_layers = num_layers
        self.encoder = encoder
        self.use_gru = use_gru

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])

        # Acceleration heads
        self.acc_mean_head = nn.Linear(hidden_size, 1)
        self.acc_sde_head = nn.Linear(hidden_size, 1)
        # self.acc_sde = nn.Parameter(torch.ones(1))

        # Steering heads
        self.ste_mean_head = nn.Linear(hidden_size, 1)
        self.ste_sde_head = nn.Linear(hidden_size, 1)
        # self.ste_sde = nn.Parameter(torch.ones(1))

        self.to(device)

    def forward(self, x, a, h_0=None, partial_forward=True):
        in_x = x
        if self.use_gru:
            h_0, x = self.encoder(x, a, h_0)
            x = x.squeeze(0)
        else:
            x = self.encoder(x, a, partial_forward)

        x = x/torch.norm(x, p=2, dim=-1, keepdim=True)
        # x = torch.cat([x, in_x], dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        acc_mean = self.acc_mean_head(x)
        acc_sde = self.acc_sde_head(x)
        # acc_sde = self.acc_sde

        ste_mean = torch.tanh(self.ste_mean_head(x))
        ste_sde = self.ste_sde_head(x)
        # ste_sde = self.ste_sde

        return h_0, (acc_mean, acc_sde), (ste_mean, ste_sde)
    
class LinearModel(nn.Module):
    def __init__(self, in_size, out_size, device, num_hidden=1, encoder=None, use_gru=True):
        super(LinearModel, self).__init__()
        self.encoder = encoder

        self.use_gru = use_gru
        self.in_size = in_size

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, in_size) for i in range(num_hidden)])

        self.linear = nn.Linear(in_size, out_size)
        self.to(device)

    def forward(self, x, h_0=None, partial_forward=True):
        output = None
        prop = x['prop']
        item_and_utility = x['item_and_utility']
        if self.use_gru:
            output, x = self.encoder(prop, h_0)
            x = x.squeeze(0)
        else:
            x = self.encoder(prop, partial_forward)

        x = x/torch.linalg.norm(x)

        if partial_forward:
            x = torch.cat([
                item_and_utility,
                x
            ], dim=-1)
        else:
            x = torch.cat([
                item_and_utility.permute((1, 0, 2)),
                x
            ], dim=-1)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return output, self.linear(x)

class F1LinearModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device, num_hidden=1, encoder=None, use_gru=True):
        super(F1LinearModel, self).__init__()
        self.encoder = encoder

        self.use_gru = use_gru
        self.in_size = in_size

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_hidden)])

        self.linear = nn.Linear(hidden_size, out_size)
        self.to(device)

    def forward(self, x, a=None, h_0=None, partial_forward=True, all_representations=False):
        in_x = x
        output = None
        if self.use_gru:
            output, x = self.encoder(x, h_0)
            x = x.squeeze(0)
        else:
            x = self.encoder(x, a, partial_forward)
            if not all_representations:
                x = x[:, 1::2, :]
        
        x = x/torch.linalg.norm(x)
        # x = torch.cat([x, in_x], dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return output, self.linear(x)
    