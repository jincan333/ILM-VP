'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''

import os 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders']

def cifar10_dataloaders(batch_size=128, data_dir='data/cifar10', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), padding=96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), padding=96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t')
    print('5000 images for testing\t')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir='data/cifar100', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), padding=96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
    ])

    test_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), padding=96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
