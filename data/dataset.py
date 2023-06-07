'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''

import os 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders']

# Imagenet Transform
def image_transform(args):
    IMAGENETNORMALIZE = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
    CIFAR10NORMALIZE = {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    }
    
    if args.prompt_method:
        normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
        if args.randomcrop:
            print('Using randomcrop\n')
            train_transform = transforms.Compose([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=CIFAR10NORMALIZE['mean'], std=CIFAR10NORMALIZE['std'])
        train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform, test_transform, normalize

def cifar10_dataloaders(args):

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t')
    data_dir = os.path.join(args.data, args.dataset)
    train_transform, test_transform, normalize = image_transform(args)
    
    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    configs = {
        'class_names': test_set.classes,
        'normalize': normalize
    }
    return train_loader, val_loader, test_loader, configs

def cifar100_dataloaders(batch_size=128, data_dir='dataset/cifar100', num_workers=2):

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
