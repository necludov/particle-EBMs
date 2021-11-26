from copy import deepcopy
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import distributions
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.utils.spectral_norm import SpectralNorm
import torchvision.transforms as tr
import torchvision as tv
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

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def eval_classification(ebm, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = ebm.label_log_prob(x_p_d)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y_p_d).detach().cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().detach().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss

def get_data(args):
    if args.logit:
        transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.), logit])
    else:
        transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.)])
    if args.data == "mnist":
        dset_train = tv.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
        dset_test = tv.datasets.MNIST(root="../data", train=False, transform=transform, download=True)
    elif args.data == "fashionmnist":
        dset_train = tv.datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
        dset_test = tv.datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)
    else:
        assert False, "wrong dataset"

    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dload_train, dload_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['mnist', 'fashionmnist'], type=str, default='mnist')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logit', action="store_true")
    parser.add_argument('--relu', action="store_true")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_particles', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ld_step', type=float, default=1e0)
    parser.add_argument('--ld_sigma', type=float, default=0.005)
    parser.add_argument('--ld_n_iter', type=int, default=50)
    args = parser.parse_args()
    
    if args.data == "mnist" or args.data == "fashionmnist":
        args.data_dim = 784
        args.data_shape = (1, 28, 28)
    
    target_name = args.data
    exp_name = 'baseline_conditional_' + target_name
    path = './logs/' + exp_name + '.pt'
    logger = Logger(exp_name, fmt={'lr': '.2e', 'loss': '.4e', 'mmd': '.4e'})
        
    batch_size = args.batch_size
    n_particles = args.n_particles
    
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    
    mmd = MMDStatistic(batch_size, n_particles)
    logger.print('Starting experiment with seed={}'.format(args.seed))
    logger.print('device:', device)
    
    dload_train, dload_test = get_data(args)
    for x, _ in dload_train:
        init_batch = x.view(x.size(0), -1).to(device)
        break

    mu, sigma = init_batch.mean(), init_batch.std()
    print('mu:', mu.cpu().numpy())
    print('sigma:', sigma.cpu().numpy())
    base_dist = distributions.Normal(mu, sigma)
    net = networks.MNISTSmallConvNet(nc=64, n_out=10, relu=args.relu)
    ebm = models.EBMC(net, base_dist=None).to(device)
    previous_ebm = ebm.get_copy()
    particles = base_dist.sample((n_particles,args.data_dim)).view(-1, *args.data_shape)
    particles = torch.randn_like(particles)
    particles = ebm.sample(particles, dt=args.ld_step, sigma=args.ld_sigma, n_steps=100)
    
    rb = ReplayBuffer(int(1e4), device)
    rb.add_samples(particles)
    label_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, betas=(.0, .999))
    t = 0
    for epoch in range(args.epochs):
        for batch_id, (x, y) in enumerate(dload_train):
            train_batch = x.to(device).view(-1, *args.data_shape)
            y = y.to(device)
            logger.add_scalar(t, 'lr', optimizer.param_groups[0]['lr'])
            
            #sample particles
            particles = rb.get_samples(n_particles)
            resample_mask = torch.zeros(args.n_particles).bool()
            resample_mask[:int(args.n_particles*0.1)] = 1.0
            resample_mask = resample_mask[torch.randperm(len(resample_mask))]
            particles[resample_mask] = torch.randn_like(particles)[resample_mask]
            particles = ebm.sample(particles, dt=args.ld_step, sigma=args.ld_sigma, n_steps=args.ld_n_iter)
            rb.add_samples(particles)
            
            #update model
            for p in ebm.parameters(): p.grad = None
            loss = ebm(train_batch).mean()-ebm(particles).mean() + label_loss(ebm.label_log_prob(train_batch),y)
            loss.backward()
            logger.add_scalar(t, 'loss', loss.detach().cpu().numpy())
            optimizer.step()
            
            mmd_value = mmd(particles.view(particles.size(0), -1), 
                            train_batch.view(particles.size(0), -1), 
                            1e-4*np.ones(args.data_dim))
            logger.add_scalar(t, 'mmd', mmd_value.detach().cpu().numpy())
            if (t % args.save_period) == 0:
                acc, cross_ent = eval_classification(ebm, dload_test, device)
                logger.add_scalar(t, 'acc', acc)
                logger.add_scalar(t, 'cross_ent', cross_ent)
                logger.save()
                torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
                torch.save(particles, './checkpoints/particles_' + logger.name)
            logger.iter_info()
            t += 1
    logger.save()
    torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
    torch.save(particles, './checkpoints/particles_' + logger.name)
