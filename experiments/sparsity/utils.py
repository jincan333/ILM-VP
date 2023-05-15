'''
    setup model and datasets
'''
import copy 
import torch
import torch.nn as nn
import torchvision
import numpy as np 
# from advertorch.utils import NormalizeByChannelMeanStd

# from models import *
import sys
sys.path.append('C:/Users/jinca/Documents/Projects/ILM-VP/data')
from dataset import *
__all__ = ['setup_model_dataset']



class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'cifar100':
        classes = 100
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    else:
        raise ValueError('Dataset not supprot yet !')

    # if args.imagenet_arch:
    #     model = model_dict[args.arch](num_classes=classes, imagenet=True)
    # else:
    #     model = model_dict[args.arch](num_classes=classes)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, classes)
    print(model)

    return model, train_set_loader, val_loader, test_loader


