import torch

import numpy as np
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

class GruModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1):
        super(GruModel, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(in_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.to(device)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        x = F.relu(self.linear(x))
        output, x = self.gru(x.unsqueeze(1), h_0)
        return output, x

class MLPModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, gru=None):
        super(MLPModel, self).__init__()
        self.gru = gru
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

    def forward(self, x, h_0=None):
        output, x = self.gru(x, h_0)
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
    
class LinearModel(nn.Module):
    def __init__(self, in_size, out_size, device, gru=None):
        super(LinearModel, self).__init__()
        self.gru = gru
        self.linear = nn.Linear(in_size, out_size)
        self.to(device)

    def forward(self, x, h_0=None):
        output, x = self.gru(x, h_0)
        return output, self.linear(x)
    