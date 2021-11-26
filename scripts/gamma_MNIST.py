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
import torchvision.transforms as tr
import torchvision as tv
import sys
sys.path.append('./')
from core import networks
from core import toy_data
from core import models
from utils.utils import MMDStatistic
from utils.logger import Logger

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def get_data(args):
    if args.logit:
        transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.), logit])
#         transform = tr.Compose([tr.ToTensor(), logit])
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
    parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_particles', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ld_step', type=float, default=1e-3)
    parser.add_argument('--ld_sigma', type=float, default=np.sqrt(1e-3))
    parser.add_argument('--ld_n_iter', type=int, default=100)
    args = parser.parse_args()
    
    if args.data == "mnist" or args.data == "fashionmnist":
        args.data_dim = 784
        args.data_shape = (1, 28, 28)
    
    target_name = args.data
    exp_name = 'gammaConv_' + target_name
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
    
    particles = base_dist.sample((n_particles,args.data_dim)).view(-1, *args.data_shape)
    print('max_particles:', particles.cpu().abs().max().numpy())
    print('max_samples:', init_batch.cpu().abs().max().numpy())
    
#     net = networks.SmallConv(args.data_dim, n_c=512)
#     net = networks.SmallMLP(args.data_dim, n_hid=1000, relu=True)
#     ebm = models.EBM(net, base_dist).to(device)
#     net = networks.SmallConv(args.data_dim, n_c=128)
    net = networks.MNISTSmallConvNet(nc=64, relu=True)
    ebm_learned = models.EBM(net, base_dist).to(device)
    ebm_learned.load_state_dict(torch.load('./checkpoints/betaConvRelu_mnist', map_location=device))
    
    t = 0
#     optimizer = optim.Adam(ebm.parameters(), lr=1e-4, betas=(.0, .9))
    
    particles = torch.autograd.Variable(particles, requires_grad=True)
    optimizer = optim.Adam([particles], lr=1e0, betas=(.9, .999))
    for epoch in range(args.epochs):
        for batch_id, (x, _) in enumerate(dload_train):
            x = x.view(x.size(0), -1)
            train_batch = x.to(device).view(-1, *args.data_shape)
#             train_batch = init_batch.to(device).view(-1, *args.data_shape)

#             net = networks.SmallMLP(args.data_dim, n_hid=1000, relu=True)
            if (t % 2) == 0:
                ebm = ebm_learned
            else:
                net = networks.MNISTSmallConvNet(nc=64, relu=True)
                ebm = models.EBM(net, base_dist).to(device)
#                 ebm.load_state_dict(torch.load('./checkpoints/betaMLP_mnist', map_location=device))
#             net = networks.SmallConv(args.data_dim, n_c=256)
#             net = networks.MNISTSmallConvNet(nc=64, relu=True)
#             ebm = models.EBM(net, base_dist).to(device)
#             optimizer_ebm = optim.Adam(ebm.parameters(), lr=1e-3, betas=(.9, .999))

            #get grad (no model update)
#             for p in ebm.parameters(): p.grad = None
#             loss = ebm(train_batch).mean()-ebm(particles).mean()
#             loss.backward()
#             optimizer_ebm.step()
            
            for p in ebm.parameters(): p.grad = None
            loss = ebm(train_batch).mean()-ebm(particles).mean()
            loss.backward()
            loss_value = loss.cpu().detach().numpy()
            logger.add_scalar(t, 'loss', loss_value)

            #update particles
            diff_grad = ebm.get_grad().clone()
            for p in ebm.parameters(): p.grad = None
            def dEdt(x):
                grad_E = torch.autograd.grad(ebm(x).sum(), ebm.parameters(), retain_graph=True, create_graph=True)
                return (torch.nn.utils.parameters_to_vector(grad_E)*diff_grad).mean()

#             grad_particles = torch.autograd.Variable(particles, requires_grad=True)
#             v = torch.autograd.grad(dEdt(grad_particles), [grad_particles], retain_graph=False)[0].detach()
            
#             particles += 9e-3*v
            particles.grad = None
            loss_particles = -dEdt(particles)
            loss_particles.backward()
            optimizer.step()
            logger.add_scalar(t, 'mmd', mmd(particles.view(particles.size(0), -1), train_batch.view(particles.size(0), -1), 1e-4*np.ones(args.data_dim)).detach().cpu().numpy())
            logger.iter_info()
            if (t % args.save_period) == 0:
                logger.save()
                torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
                torch.save(particles, './checkpoints/particles_' + logger.name)
            t += 1
    logger.save()
    torch.save(ebm.state_dict(), './checkpoints/' + logger.name)
    torch.save(particles, './checkpoints/particles_' + logger.name)
