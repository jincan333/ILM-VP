import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .dataset_lmdb import COOPLMDBDataset
from .const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE


'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''

import os 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, GTSRB
import numpy as np


# Imagenet Transform
def image_transform(args):
    IMAGENETNORMALIZE = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
    normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
    if args.prompt_method:
        
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
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
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



def prepare_finetune_data(args):
    data_path = os.path.join(args.data, args.dataset)
    dataset = args.dataset
    train_transform, test_transform, normalize = image_transform(args)

    if dataset == "cifar10":
        full_data = CIFAR10(root = data_path, train = True, download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(CIFAR10(data_path, train=True, transform=train_transform, download=True), list(train_len))
        val_set = Subset(CIFAR10(data_path, train=True, transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = CIFAR10(data_path, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        configs = {
            'class_names': test_set.classes,
            'normalize': normalize
        }
        
    elif dataset == "cifar100":
        full_data = CIFAR100(root = data_path, train = True, download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(CIFAR100(data_path, train=True, transform=train_transform, download=True), list(train_len))
        val_set = Subset(CIFAR100(data_path, train=True, transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = CIFAR100(data_path, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        configs = {
            'class_names': test_set.classes,
            'normalize': normalize
        }

    elif dataset == "svhn":
        full_data = SVHN(root = data_path, split = 'train', download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(SVHN(data_path, split = 'train', transform=train_transform, download=True), list(train_len))
        val_set = Subset(SVHN(data_path, split = 'train', transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = SVHN(data_path, split = 'test', transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        class_names = [f'{i}' for i in range(10)]
        configs = {
            'class_names': class_names,
            'normalize': normalize
        }
    
    elif dataset == "gtsrb":
        full_data = GTSRB(root = data_path, split = 'train', download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), list(train_len))
        val_set = Subset(GTSRB(data_path, split = 'train', transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = GTSRB(data_path, split = 'test', transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        configs = {
            'class_names': class_names,
            'normalize': normalize
        }

    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        full_data =  COOPLMDBDataset(root = data_path, split="train")
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), list(train_len))
        val_set = Subset(GTSRB(data_path, split = 'train', transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = GTSRB(data_path, split = 'test', transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)

        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }

    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names



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






def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())


def prepare_gtsrb_fraction_data(data_path, fraction, preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction*26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        return loaders, class_names



def prepare_prompt_data(dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2, pin_memory=True),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2, pin_memory=True),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root = data_path)
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': ["non ASD", "ASD"],
            'mask': D.get_mask(),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=0, pin_memory=True),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=0, pin_memory=True),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs