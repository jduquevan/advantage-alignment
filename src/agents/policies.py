import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn

from torch.distributions.transforms import SigmoidTransform, TanhTransform

class F1NormalTanHPolicy(nn.Module):
    def __init__(self, log_std_min, log_std_max, device, model):
        super(F1NormalTanHPolicy, self).__init__()
        self.model = model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.acc_log_std = None
        self.ste_log_std = None

        self.to(device)

    def forward(self, x, h_0=None, use_tranformer=False):
        if use_tranformer:
            output, acc, ste = self.model(x, partial_forward=False)
        else:
            output, acc, ste = self.model(x, h_0)
        acc_mean, acc_log_std = acc
        ste_mean, ste_log_std = ste

        acc_log_std = torch.clamp(acc_log_std, self.log_std_min, self.log_std_max)
        ste_log_std = torch.clamp(ste_log_std, self.log_std_min, self.log_std_max)


        self.acc_log_std = acc_log_std
        self.ste_log_std = ste_log_std

        base_acc_dist = D.normal.Normal(
            loc = acc_mean.squeeze(0),
            scale = torch.exp(acc_log_std.squeeze(0))
        )
        base_ste_dist = D.normal.Normal(
            loc = ste_mean.squeeze(0),
            scale = torch.exp(ste_log_std.squeeze(0))
        )
        
        acc_normal_tanh = D.TransformedDistribution(base_acc_dist, [TanhTransform()])
        ste_normal_tanh = D.TransformedDistribution(base_ste_dist, [TanhTransform()])

        return output, acc_normal_tanh, ste_normal_tanh

    def sample_action(self, x, h_0=None, use_transformer=False):
        output, acc_normal, ste_normal = self.forward(
            x, 
            h_0, 
            use_transformer
        )
        acc = acc_normal.sample()
        ste = ste_normal.sample()

        # Necessary to prevent log probs from being nan
        acc = torch.clamp(acc, -1 + 1e-5, 1 - 1e-5)
        ste = torch.clamp(ste, -1 + 1e-5, 1 - 1e-5)

        log_p_acc = acc_normal.log_prob(acc).squeeze(1)
        log_p_ste = ste_normal.log_prob(ste).squeeze(1)
        log_p = log_p_acc + log_p_ste 

        action = np.concatenate(
            [acc.cpu().numpy(), ste.cpu().numpy()],
            axis=1
        )

        return output, action, log_p

class NormalSigmoidPolicy(nn.Module):
    def __init__(self, log_std_min, log_std_max, prop_max, device, model):
        super(NormalSigmoidPolicy, self).__init__()
        self.model = model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Upper bound on each component of proposition
        self.prop_max = prop_max

        self.utte_log_std = None
        self.prop_log_std = None

        self.to(device)

    def forward(self, x, h_0=None, use_tranformer=False):
        if use_tranformer:
            output, term_dist, utte, prop = self.model(x, partial_forward=False)
        else:
            output, term_dist, utte, prop = self.model(x, h_0)
        utte_mean, utte_log_std = utte
        prop_mean, prop_log_std = prop

        utte_log_std = torch.clamp(utte_log_std, self.log_std_min, self.log_std_max)
        prop_log_std = torch.clamp(prop_log_std, self.log_std_min, self.log_std_max)

        if use_tranformer:
            term_dist = term_dist.permute((1, 0, 2))
            utte_mean = utte_mean.permute((1, 0, 2))
            prop_mean = prop_mean.permute((1, 0, 2))
            utte_log_std = utte_log_std.permute((1, 0, 2))
            prop_log_std = prop_log_std.permute((1, 0, 2))

        self.utte_log_std = utte_log_std
        self.prop_log_std = prop_log_std

        # TODO: Check for batch support
        base_term_dist = D.Categorical(term_dist)
        base_utte_dist = D.normal.Normal(
            loc = utte_mean.squeeze(0),
            scale = torch.exp(utte_log_std.squeeze(0))
        )
        base_prop_dist = D.normal.Normal(
            loc = prop_mean.squeeze(0),
            scale = torch.exp(prop_log_std.squeeze(0))
        )
        
        prop_normal_sigmoid = D.TransformedDistribution(base_prop_dist, [SigmoidTransform()])
        return output, base_term_dist, base_utte_dist, prop_normal_sigmoid

    def sample_action(self, x, h_0=None, use_transformer=False):
        action = {}

        output, term_categorical, utte_normal, prop_normal_sigmoid = self.forward(
            x, 
            h_0, 
            use_transformer
        )
        term = term_categorical.sample()
        utte = utte_normal.sample()
        prop = prop_normal_sigmoid.sample()

        log_p_term = term_categorical.log_prob(term).unsqueeze(0)
        log_p_utte = utte_normal.log_prob(utte).sum(dim=-1)
        log_p_prop = prop_normal_sigmoid.log_prob(prop).sum(dim=-1)  # Sum log probabilities over dimensions
        log_p = log_p_term + log_p_utte + log_p_prop

        action['term'] = term.squeeze(0).cpu().int().numpy()
        action['utte'] = utte.cpu().int().numpy()
        action['prop'] = (prop * self.prop_max).cpu().numpy()

        return output, action, log_p

class CategoricalPolicy(nn.Module):
    def __init__(self, device, model):
        super(CategoricalPolicy, self).__init__()
        self.model = model
        self.to(device)

    def forward(self, x, h_0=None, use_transformer=False, partial_forward=False):
        if use_transformer:
            output, term_dist, utte, prop = self.model(x, partial_forward=partial_forward)
        else:
            output, term_dist, utte, prop = self.model(x, h_0)

        prop_1, prop_2, prop_3 = prop

        if use_transformer and not partial_forward:
            prop_1 = prop_1.permute((1, 0, 2))
            prop_2 = prop_2.permute((1, 0, 2))
            prop_3 = prop_3.permute((1, 0, 2))
        elif not use_transformer:
            prop_1 = prop_1.squeeze(0)
            prop_2 = prop_2.squeeze(0)
            prop_3 = prop_3.squeeze(0)

        prop_dist_1 = D.Categorical(prop_1)
        prop_dist_2 = D.Categorical(prop_2)
        prop_dist_3 = D.Categorical(prop_3)
        
        return output, term_dist, utte, (prop_dist_1, prop_dist_2, prop_dist_3)

    def sample_action(self, x, h_0=None, use_transformer=False):
        action = {}

        output, term_cat, utte, prop_cat = self.forward(x, h_0, use_transformer, True)
        prop_cat_1, prop_cat_2, prop_cat_3 = prop_cat

        prop_1 = prop_cat_1.sample()
        prop_2 = prop_cat_2.sample()
        prop_3 = prop_cat_3.sample()

        prop = torch.cat([
            prop_1.unsqueeze(1), 
            prop_2.unsqueeze(1), 
            prop_3.unsqueeze(1)
        ], dim=1)

        # TODO: Add utterance support

        log_p_prop = (
            prop_cat_1.log_prob(prop_1) + 
            prop_cat_2.log_prob(prop_2) + 
            prop_cat_3.log_prob(prop_3)
        ) 
        log_p = log_p_prop

        action['term'] = term_cat
        action['utte'] = utte
        action['prop'] = prop

        return output, action, log_p
