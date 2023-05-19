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
sys.path.append(".")
from data import cifar10_dataloaders, cifar100_dataloaders
__all__ = ['setup_model_dataset']



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
    # model.fc = nn.Linear(512, classes)
    # model.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool=nn.Identity()
    print(model)

    return model, train_set_loader, val_loader, test_loader


