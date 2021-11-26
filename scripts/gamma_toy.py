from copy import deepcopy
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import distributions
from torch import nn
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import sys
sys.path.append('./')
from core import networks
from core import toy_data
from core import models
from utils.utils import MMDStatistic, log_likelihood_2d, pairwise_distances
from utils.logger import Logger

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# class ReplayBuffer:
#     def __init__(self, n_max, device):
#         self.n_max = n_max
#         self.samples = torch.tensor([], device=device)
        
#     def add_samples(self, samples):
#         self.samples = torch.cat([self.samples, samples.clone()])
#         if len(self.samples) > self.n_max:
#             ids_shuffled = torch.randperm(len(self.samples))[:self.n_max]
#             self.samples = self.samples[ids_shuffled] 
        
#     def get_samples(self, n):
#         random_ids = torch.randperm(len(self.samples))[:n]
#         return self.samples[random_ids].clone()

# class GammaE(nn.Module):
#     def __init__(self, n_max, device, init_e):
#         super(GammaE, self).__init__()
#         self.rb = ReplayBuffer(n_max, device)
#         self.init_e = init_e
#         self.device = device
#         self.t = 0.0
        
#     def add_samples(self, samples):
#         self.rb.add_samples(samples)
        
#     def forward(self, x):
#         n_samples = len(self.rb.samples)
#         e0 = self.init_e(x)
#         dE = torch.zeros(x.shape[0]).to(self.device)
#         for i in range(10):
#             net = networks.SmallMLP(2, relu=True)
#             ebm = models.EBM(net, base_dist).to(self.device)
#             samples = self.rb.get_samples(int(1e3))
#             de = ebm(samples).mean() - ebm(train_batch).mean()
#             de.backward()
#             diff_grad = ebm.get_grad().clone()
#             for p in ebm.parameters(): p.grad = None
#             z = torch.autograd.Variable(torch.ones(x.shape[0]).to(self.device), requires_grad=True)
#             grad_E = torch.autograd.grad((ebm(x)*z).sum(), ebm.parameters(), retain_graph=True, create_graph=True)
#             dE += torch.autograd.grad((torch.nn.utils.parameters_to_vector(grad_E)*diff_grad).sum(), z)[0].detach()/10.0
#         return e0 + self.t*dE
    
#     def get_device(self):
#         return self.device

class GammaE(nn.Module):
    def __init__(self, device, init_e):
        super(GammaE, self).__init__()
        grid_size = 200
        x_min, x_max = -5., 5.
        y_min, y_max = -5., 5.
        self.dx, self.dy = (y_max-y_min)/grid_size, (x_max-x_min)/grid_size
        x = torch.linspace(x_min,x_max,grid_size).to(device)
        y = torch.linspace(x_min,x_max,grid_size).to(device)
        x_grid,y_grid = torch.meshgrid(x,y)
        self.points = torch.stack([x_grid, y_grid], axis=2).reshape([-1, 2])
        self.E = init_e(self.points)
        self.device = device
        
    def dedt(self, ebm, samples, train_batch, dt):
        for p in ebm.parameters(): p.grad = None
        loss = ebm(samples).mean() - ebm(train_batch).mean()
        loss.backward()
        diff_grad = ebm.get_grad().clone()
        for p in ebm.parameters(): p.grad = None
        z = torch.autograd.Variable(torch.ones(self.points.shape[0]).to(self.device), requires_grad=True)
        grad_E = torch.autograd.grad((ebm(self.points)*z).sum(), ebm.parameters(), retain_graph=True, create_graph=True)
        dE = torch.autograd.grad((torch.nn.utils.parameters_to_vector(grad_E)*diff_grad).sum(), z)[0].detach()
        self.E += dt*dE
        
    def get_logl(self, train_batch):
        pd = pairwise_distances(train_batch, self.points)
        dists, ids = torch.min(pd, dim=1)
        logZ = torch.logsumexp(-self.E,0) + np.log(self.dx) + np.log(self.dy)
        return -self.E[ids].mean()-logZ

def sample_data(name, batch_size, rng):
    x = toy_data.inf_train_gen(name, rng, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    return x

def dEdt(x, ebm, diff_grad):
    grad_E = torch.autograd.grad(ebm(x).sum(), ebm.parameters(), retain_graph=True, create_graph=True)
    return torch.nn.utils.parameters_to_vector(grad_E)*diff_grad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons',
                 '2spirals', 'checkerboard', 'rings'],
        type=str, default='8gaussians'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=2e-2)
    args = parser.parse_args()
    
    target_name = args.data
    exp_name = 'gamma_' + target_name + ('_seed_%d' % args.seed)
    path = './logs/' + exp_name + '.pt'
    logger = Logger(exp_name, fmt={'lr': '.2e', 'loss': '.4e', 'mmd': '.4e'})
        
    batch_size = args.batch_size
    n_particles = args.n_particles
    
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    
    mmd = MMDStatistic(batch_size, n_particles)
    logger.print('Starting experiment with seed={}'.format(args.seed))
    logger.print('device:', device)
    train_batch = sample_data(target_name, batch_size, rng)

    mu, sigma = train_batch.mean(0), train_batch.std(0)
    base_dist = distributions.Normal(mu, sigma)
    
    particles = base_dist.sample((n_particles,))
    gamma_energy = GammaE(device, lambda x: -base_dist.log_prob(x).view(x.size(0), -1).sum(1))
    for t in range(args.n_iters):
        net = networks.SmallMLP(2, relu=True)
        ebm = models.EBM(net, base_dist).to(device)
        
        train_batch = sample_data(target_name, batch_size, rng)
        gamma_energy.dedt(ebm, particles, train_batch, args.dt)
        
        #get grad (no model update)
        for p in ebm.parameters(): p.grad = None
        loss = ebm(train_batch).mean()-ebm(particles).mean()
        loss.backward()
        logger.add_scalar(t, 'loss', loss.detach().cpu().numpy())
        logl = gamma_energy.get_logl(train_batch)
        logger.add_scalar(t, 'logl', logl.detach().cpu().numpy())
        
        #update particles
        diff_grad = ebm.get_grad().clone()
        for p in ebm.parameters(): p.grad = None
        grad_particles = torch.autograd.Variable(particles, requires_grad=True)
        particles += args.dt*torch.autograd.grad(dEdt(grad_particles, ebm, diff_grad).sum(),
                                                 [grad_particles], retain_graph=True)[0].detach()
        logger.add_scalar(t, 'mmd', mmd(particles, train_batch, np.ones(2)).detach().cpu().numpy())
        logger.iter_info()
        if (t % args.save_period) == 0:
            logger.save()
            torch.save(gamma_energy.E, './checkpoints/' + logger.name)
            torch.save(particles, './checkpoints/particles_' + logger.name)
    logger.save()
    torch.save(gamma_energy.E, './checkpoints/' + logger.name)
    torch.save(particles, './checkpoints/particles_' + logger.name)
