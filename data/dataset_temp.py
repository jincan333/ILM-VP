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
        1. CIFAR-10
        2. CIFAR-100   
        3. SVHN
        4. GTSRB
        5. FOOD-101
        6. SUN-397
        7. EUROSAT
        8. UCF-101
        9. Stanford Cars
        10. FLOWERS-102
        11. DTD
        12. Oxford Pets
'''

import os 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, GTSRB, Food101, SUN397, EuroSAT, UCF101, StanfordCars, Flowers102, DTD, OxfordIIITPet
import numpy as np


# Imagenet Transform
def image_transform(args):
    normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
    if args.prompt_method:
        
        if args.randomcrop and args.dataset=='cifar10':
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


def prepare_dataset(args):
    data_path = os.path.join(args.data, args.dataset)
    dataset = args.dataset
    train_transform, test_transform, normalize = image_transform(args)

    if dataset == "cifar10":
        full_data = CIFAR10(root = data_path, train = True, download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(CIFAR10(data_path, train=True, transform=train_transform, download=True), list(range(train_len)))
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
        train_set = Subset(CIFAR100(data_path, train=True, transform=train_transform, download=True), list(range(train_len)))
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
        # indices = np.random.permutation(num_samples)

        # # Split the shuffled indices
        # train_size = int(train_ratio * num_samples)
        # train_indices = indices[:train_size]
        # val_indices = indices[train_size:]
        train_set = Subset(SVHN(data_path, split = 'train', transform=train_transform, download=True), list(range(train_len)))
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
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), list(range(train_len)))
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
        train_set = Subset(COOPLMDBDataset(data_path, split = 'train', transform=train_transform, download=True), list(range(train_len)))
        val_set = Subset(COOPLMDBDataset(data_path, split = 'train', transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = COOPLMDBDataset(data_path, split = 'test', transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
        class_names = refine_classnames(test_set.classes)
        configs = {
            'class_names': class_names,
            'normalize': normalize
        }

    elif dataset in ["dtd", "oxfordpets"]:
        full_data =  COOPLMDBDataset(root = data_path, split="train")
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(COOPLMDBDataset(data_path, split = 'train', transform=train_transform, download=True), list(range(train_len)))
        val_set = Subset(COOPLMDBDataset(data_path, split = 'train', transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = COOPLMDBDataset(data_path, split = 'test', transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
        class_names = refine_classnames(test_set.classes)
        configs = {
            'class_names': class_names,
            'normalize': normalize
        }

    else:
        raise NotImplementedError(f"{dataset} not supported")
    
    print(f'Dataset information: {dataset}\t {train_len} images for training \t {full_len - train_len} images for validation\t')
    print(f'{len(test_set)} images for testing\t')

    return train_loader, val_loader, test_loader, configs


def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())
