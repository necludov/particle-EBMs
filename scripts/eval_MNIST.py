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

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def get_data(args):
    if args.logit:
        transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.), logit])
    else:
        transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.)])
    if args.data == "mnist":
        dset_train = tv.datasets.MNIST(root="../data", train=True, transform=transform, download=False)
        dset_test = tv.datasets.MNIST(root="../data", train=False, transform=transform, download=False)
    elif args.data == "fashionmnist":
        dset_train = tv.datasets.FashionMNIST(root="../data", train=True, transform=transform, download=False)
        dset_test = tv.datasets.FashionMNIST(root="../data", train=False, transform=transform, download=False)
    else:
        assert False, "wrong dataset"

    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dload_train, dload_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['mnist', 'fashionmnist'], type=str, default='mnist')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logit', action="store_true")
    parser.add_argument('--relu', action="store_true")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_period', type=int, default=5000)
    parser.add_argument('--print_period', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ld_step', type=float, default=1e-1)
    parser.add_argument('--ld_sigma', type=float, default=np.sqrt(1e-1))
    parser.add_argument('--ld_n_iter', type=int, default=10)
    args = parser.parse_args()
    
    args.data_dim = 784
    args.data_shape = (1, 28, 28)
    
    exp_name = 'eval_'
    if args.checkpoint is not None:
        exp_name += '_' + args.checkpoint.split('/')[-1]
    else:
        raise AssertionError('define checkpoint')
    path = './logs/' + exp_name + '.pt'
    logger = Logger(exp_name, fmt={'AR': '.3f', '-logZ': '.4e'})
    
    batch_size = args.batch_size
    n_particles = args.n_particles
    
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    
    net = networks.MNISTSmallConvNet(nc=64, relu=True)
    ebm = models.EBM(net, base_dist=None).to(device)
    ebm.load_state_dict(torch.load(args.checkpoint, map_location=device))

    logger.print('Starting experiment with seed={}'.format(args.seed))
    logger.print('device:', device)
    
    dt = 1e-2
    sigma = np.sqrt(dt)
    n_samples = n_particles
    n_iter = 100000
    data_shape = (1, 28, 28)
    data_dim = 28*28
    beta_schedule = np.linspace(0.0,1.0, n_iter)

    init_dist = distributions.Normal(torch.zeros(1).sum().to(device), torch.ones(1).sum().to(device))
    samples = init_dist.sample([n_samples,*data_shape]).view(-1,*data_shape).to(device)
    samples = torch.autograd.Variable(samples, requires_grad=True)
    log_w = torch.zeros(n_samples)
    log_prob = lambda x, beta: (beta*(-ebm(x))+(1-beta)*init_dist.log_prob(x.view(-1,data_dim)).sum(1))
    acceptance_rate = 0.0
    for i in range(1,n_iter):
        beta, beta_prev = beta_schedule[i], beta_schedule[i-1]
        with torch.no_grad():
            log_w += (log_prob(samples,beta) - log_prob(samples,beta_prev)).cpu()
        grad = torch.autograd.grad(log_prob(samples, beta).sum(), samples)[0].detach()
        proposals = samples.data + dt/2. * grad + torch.randn_like(samples) * sigma
        proposals = torch.autograd.Variable(proposals, requires_grad=True)
        grad_p = torch.autograd.grad(log_prob(proposals, beta).sum(), proposals)[0].detach()
        with torch.no_grad():
            log_P = log_prob(proposals.data, beta)-log_prob(samples.data, beta)
            log_P += - 0.5/sigma**2*((samples-(proposals+dt/2.*grad_p))**2).view(-1, data_dim).sum(1)
            log_P += + 0.5/sigma**2*((proposals-(samples+dt/2.*grad))**2).view(-1, data_dim).sum(1)
            log_U = torch.log(torch.rand_like(log_P))
        accept_mask = log_P > log_U
        acceptance_rate += accept_mask.float().mean().cpu()
        samples.data[accept_mask] = proposals.data[accept_mask]
        if (i % args.save_period) == 0:
            logger.save()
            torch.save(samples.data, ('./checkpoints/samples_%d_' % i) + logger.name)
        if (i % args.print_period) == 0:
            logger.add_scalar(i, 'AR', acceptance_rate.numpy()/args.print_period)
            acceptance_rate = 0.0
            logger.add_scalar(i, '-logZ', -torch.logsumexp(log_w,0).numpy()+np.log(n_samples))
            logger.iter_info()
    logZ = torch.logsumexp(log_w,0).numpy()-np.log(n_samples)
    dload_train, dload_test = get_data(args)
    log_p = 0.0
    for batch_id, (x, _) in enumerate(dload_train):
        train_batch = x.to(device).view(-1, *args.data_shape)
        with torch.no_grad():
            log_p += -ebm(train_batch).mean()/len(dload_train)
    logger.print('average -E: %.5e' % log_p)        
    log_p += -logZ
    logger.print('log-likelihood: %.5e' % log_p)
    logger.save()
    torch.save(samples.data, ('./checkpoints/samples_%d_' % i) + logger.name)
