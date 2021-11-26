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

def sample_data(name, batch_size, rng):
    x = toy_data.inf_train_gen(name, rng, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    return x

def get_vector_field(ebm, train_batch, particles):
    ebm_data = deepcopy(ebm)
    ebm_data_sqr = deepcopy(ebm)
    
    energies_data = ebm(train_batch).detach().view([-1,1])
    v = torch.zeros_like(particles)
    for i in range(v.shape[0]):
        for p in ebm.parameters(): p.grad = None
        for p in ebm_data.parameters(): p.grad = None
        for p in ebm_data_sqr.parameters(): p.grad = None
        grad_batch = torch.autograd.Variable(train_batch, requires_grad=True)
        grad_batch_sqr = torch.autograd.Variable(train_batch, requires_grad=True)

        grad_theta_data = torch.autograd.grad(ebm_data(grad_batch).sum(), ebm_data.parameters(), 
                                              retain_graph=True, create_graph=True)
        grad_theta_data_sqr = torch.autograd.grad((ebm_data_sqr(grad_batch_sqr)**2).sum(), ebm_data_sqr.parameters(), 
                                                  retain_graph=True, create_graph=True)
        grad_theta_particle = torch.autograd.grad(ebm(particles[i].view([1,-1])).sum(), ebm.parameters())
        scal = (nn.utils.parameters_to_vector(grad_theta_data)*nn.utils.parameters_to_vector(grad_theta_particle)).sum()
        v_1 = torch.autograd.grad(scal, grad_batch, retain_graph=True)[0].detach()
        v_1 = (v_1*(-energies_data-1)).mean(0)
        scal_sqr = (nn.utils.parameters_to_vector(grad_theta_data_sqr)*nn.utils.parameters_to_vector(grad_theta_particle)).sum()
        v_2 = torch.autograd.grad(scal_sqr, grad_batch_sqr, retain_graph=True)[0].detach()
        v_2 = 0.5*v_2.mean(0)
        v[i] = v_1 + v_2
    return v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons',
                 '2spirals', 'checkerboard', 'rings'],
        type=str, default='8gaussians'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ld_step', type=float, default=1e-2)
    parser.add_argument('--ld_sigma', type=float, default=np.sqrt(1e-2))
    parser.add_argument('--ld_n_iter', type=int, default=10)
    args = parser.parse_args()
    
    target_name = args.data
    exp_name = 'delta_' + target_name
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
    previous_ebm = deepcopy(ebm)
    
    particles = ebm.sample(base_dist.sample((n_particles,)), dt=1e-4, sigma=np.sqrt(1e-4), n_steps=1000)
    
    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, betas=(.0, .999))
    for t in range(args.n_iters):
        train_batch = sample_data(target_name, batch_size, rng)
        logger.add_scalar(t, 'lr', optimizer.param_groups[0]['lr'])
        
        #update model
        for p in ebm.parameters(): p.grad = None
        loss = ebm(train_batch).mean()-ebm(particles).mean()
        loss.backward()
        logger.add_scalar(t, 'loss', loss.detach().cpu().numpy())
        optimizer.step()
        logl = log_likelihood_2d(ebm, train_batch)
        logger.add_scalar(t, 'logl', logl.detach().cpu().numpy())
        
        #update particles
        for p in ebm.parameters(): p.grad = None
        v = get_vector_field(ebm, train_batch, particles)
        
        particles += 1e-3*v
        particles = ebm.sample(particles, dt=args.ld_step, sigma=args.ld_sigma, n_steps=args.ld_n_iter)
        logger.add_scalar(t, 'mmd', mmd(particles, train_batch, 0.1*np.ones(2)).detach().cpu().numpy())
        logger.iter_info()
        if (t % args.save_period) == 0:
            logger.save()
            torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
            torch.save(particles, './checkpoints/particles_' + logger.name)
    logger.save()
    torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
    torch.save(particles, './checkpoints/particles_' + logger.name)
