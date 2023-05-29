'''
    setup model and datasets
'''
import torch
import torchvision
import os
import sys
import numpy as np
import random
sys.path.append(".")
from data import cifar10_dataloaders, cifar100_dataloaders
__all__ = ['setup_model_dataset']


def setup_model_dataset(args):
    data_dir = os.path.join(args.data, args.dataset)
    if args.dataset == 'cifar10':
        train_set_loader, val_loader, test_loader, configs = cifar10_dataloaders(args)
    elif args.dataset == 'cifar100':
        train_set_loader, val_loader, test_loader, configs = cifar100_dataloaders(batch_size = args.batch_size, data_dir = data_dir, num_workers = args.workers)
    else:
        raise ValueError('Dataset not supprot yet !')
    
    if args.network == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.cuda()
    return model, train_set_loader, val_loader, test_loader, configs


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True