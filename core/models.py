##############################################################
# Code from https://github.com/wgrathwohl/LSD
##############################################################

import numpy as np
import torch
from torch import nn
from torch import distributions
from copy import deepcopy
from torch.nn.utils.spectral_norm import SpectralNorm

class EBM(nn.Module):
    def __init__(self, net, base_dist=None, learn_base_dist=True):
        super(EBM, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc, requires_grad=learn_base_dist)
            self.base_logstd = nn.Parameter(base_dist.scale.log(), requires_grad=learn_base_dist)
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = -base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        net = self.net(x)
        return net + bd

    def sample(self, init_samples, dt=0.01, sigma=np.sqrt(0.01), n_steps=100, anneal=None, clip_value=None):
        samples = torch.autograd.Variable(init_samples.detach().clone(), requires_grad=True)
        if anneal == "lin":
            dts = np.linspace(dt, sigma**2, n_steps)
        elif anneal == "log":
            dts = np.logspace(np.log10(dt), np.log10(sigma**2), n_steps)
        else:
            dts = [dt for _ in range(n_steps)]
        for this_dt in dts:
            nablaE = torch.autograd.grad(self(samples).sum(), [samples], retain_graph=True)[0]
            if clip_value is not None:
                nablaE = torch.clip(nablaE, min=-clip_value, max=clip_value)
            samples.data += this_dt/2. * (-nablaE) + torch.randn_like(samples) * sigma
        final_samples = samples.detach()
        return final_samples
    
    def get_params(self):
        return torch.nn.utils.parameters_to_vector(self.parameters())
    
    def get_grad(self):
        return torch.cat([param.grad.view(-1) for param in self.parameters()])
    
    def get_device(self):
        return next(self.net.parameters()).device
    
    def get_copy(self):
        has_hooks = False
        for module in self.net.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, SpectralNorm):
                    has_hooks = True
                    delattr(module, hook.name)
        copy_ebm = deepcopy(self)
        if has_hooks:
            for module in self.net.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, SpectralNorm):
                        hook(module, None)
            for module in copy_ebm.net.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, SpectralNorm):
                        hook(module, None)
        return copy_ebm
    
class EBMC(nn.Module):
    def __init__(self, net, base_dist=None, learn_base_dist=True):
        super(EBMC, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc, requires_grad=learn_base_dist)
            self.base_logstd = nn.Parameter(base_dist.scale.log(), requires_grad=learn_base_dist)
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = -base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        cond_energy = self.net(x)
        total_energy = -torch.logsumexp(-cond_energy, 1) + bd
        return total_energy
    
    def label_log_prob(self, x):
        return -self.net(x)

    def sample(self, init_samples, dt=0.01, sigma=np.sqrt(0.01), n_steps=100, anneal=None, clip_value=None):
        samples = torch.autograd.Variable(init_samples.detach().clone(), requires_grad=True)
        if anneal == "lin":
            dts = np.linspace(dt, sigma**2, n_steps)
        elif anneal == "log":
            dts = np.logspace(np.log10(dt), np.log10(sigma**2), n_steps)
        else:
            dts = [dt for _ in range(n_steps)]
        for this_dt in dts:
            nablaE = torch.autograd.grad(self(samples).sum(), [samples], retain_graph=True)[0]
            if clip_value is not None:
                nablaE = torch.clip(nablaE, min=-clip_value, max=clip_value)
            samples.data += this_dt/2. * (-nablaE) + torch.randn_like(samples) * sigma
        final_samples = samples.detach()
        return final_samples
    
    def get_params(self):
        return torch.nn.utils.parameters_to_vector(self.parameters())
    
    def get_grad(self):
        return torch.cat([param.grad.view(-1) for param in self.parameters()])
    
    def get_device(self):
        return next(self.net.parameters()).device
    
    def get_copy(self):
        has_hooks = False
        for module in self.net.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, SpectralNorm):
                    has_hooks = True
                    delattr(module, hook.name)
        copy_ebm = deepcopy(self)
        if has_hooks:
            for module in self.net.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, SpectralNorm):
                        hook(module, None)
            for module in copy_ebm.net.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, SpectralNorm):
                        hook(module, None)
        return copy_ebm