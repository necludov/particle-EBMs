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
from utils.utils import MMDStatistic, log_likelihood_2d
from utils.logger import Logger

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, n_max, device):
        self.n_max = n_max
        self.samples = torch.tensor([], device=device)
        
    def add_samples(self, samples):
        self.samples = torch.cat([self.samples, samples.clone()])
        if len(self.samples) > self.n_max:
            ids_shuffled = torch.randperm(len(self.samples))[:self.n_max]
            self.samples = self.samples[ids_shuffled] 
        
    def get_samples(self, n):
        random_ids = torch.randperm(len(self.samples))[:n]
        return self.samples[random_ids].clone()

def sample_data(name, batch_size, rng):
    x = toy_data.inf_train_gen(name, rng, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    return x

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
    parser.add_argument('--resample_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ld_step', type=float, default=1e-1)
    parser.add_argument('--ld_sigma', type=float, default=np.sqrt(1e-2))
    parser.add_argument('--ld_n_iter', type=int, default=20)
    parser.add_argument('--ld_clip', type=float, default=None)
    args = parser.parse_args()
    
    target_name = args.data
    exp_name = 'baseline_' + target_name + ('_seed_%d' % args.seed)
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

    net = networks.SmallMLP(2)
    mu, sigma = train_batch.mean(0), train_batch.std(0)
    base_dist = distributions.Normal(mu, sigma)
    ebm = models.EBM(net, base_dist).to(device)
    
    particles = ebm.sample(base_dist.sample((n_particles,)), dt=1e-2, 
                           sigma=np.sqrt(1e-2), n_steps=1000)
    
    rb = ReplayBuffer(int(1e4), device)
    rb.add_samples(particles)
    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, betas=(.0, .999))
    for t in range(args.n_iters):
        train_batch = sample_data(target_name, batch_size, rng)
        logger.add_scalar(t, 'lr', optimizer.param_groups[0]['lr'])
        
        #sample particles
        particles = rb.get_samples(n_particles)
        resample_mask = torch.rand(n_particles) < args.resample_rate
        particles[resample_mask] = base_dist.sample((resample_mask.sum(),))
        particles = ebm.sample(particles, dt=args.ld_step, sigma=args.ld_sigma, 
                               n_steps=args.ld_n_iter, clip_value=args.ld_clip)
        rb.add_samples(particles)
        logger.add_scalar(t, 'mmd', mmd(particles, train_batch, np.ones(2)).detach().cpu().numpy())
        
        #update model
        for p in ebm.parameters(): p.grad = None
        loss = ebm(train_batch).mean()-ebm(particles).mean()
        loss.backward()
        optimizer.step()
        logger.add_scalar(t, 'loss', loss.detach().cpu().numpy())
        logl = log_likelihood_2d(ebm, train_batch)
        logger.add_scalar(t, 'logl', logl.detach().cpu().numpy())
        if (t % args.save_period) == 0:
            logger.save()
            torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
            torch.save(particles, './checkpoints/particles_' + logger.name)
        logger.iter_info()
    logger.save()
    torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
    torch.save(particles, './checkpoints/particles_' + logger.name)
