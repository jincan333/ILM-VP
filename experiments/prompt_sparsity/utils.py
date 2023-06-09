'''
    setup model and datasets
'''
import torch
import torchvision
import os
import sys
import numpy as np
import random
from functools import partial


sys.path.append(".")
from data import cifar10_dataloaders, cifar100_dataloaders, prepare_dataset
from models import ExpansiveVisualPrompt, PadVisualPrompt, FixVisualPrompt, RandomVisualPrompt
from algorithms import label_mapping_base, generate_label_mapping_by_frequency_ordinary, generate_label_mapping_by_frequency
__all__ = ['setup_model_dataset', 'set_seed', 'setup_optimizer_scheduler', 'calculate_label_mapping', 'obtain_label_mapping']


def get_model(args):
    # network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(args.device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    
    return network

def setup_model_dataset(args):
    data_dir = os.path.join(args.data, args.dataset)
    if args.dataset == 'cifar10':
        train_set_loader, val_loader, test_loader, configs = cifar10_dataloaders(args)
    elif args.dataset == 'cifar100':
        train_set_loader, val_loader, test_loader, configs = cifar100_dataloaders(batch_size = args.batch_size, data_dir = data_dir, num_workers = args.workers)
    else:
        raise ValueError('Dataset not supprot yet !')
    
    # network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(args.device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    
    return network, train_set_loader, val_loader, test_loader, configs


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_optimizer_scheduler(network, configs, device, args):
    if args.prompt_method:
        if args.prompt_method == 'expand':
            visual_prompt = ExpansiveVisualPrompt(args, normalize=configs['normalize']).to(device)
        elif args.prompt_method == 'pad':
            visual_prompt = PadVisualPrompt(args, normalize=configs['normalize']).to(device)
        elif args.prompt_method == 'fix':
            visual_prompt = FixVisualPrompt(args, normalize=configs['normalize']).to(device)
        elif args.prompt_method == 'random':
            visual_prompt = RandomVisualPrompt(args, normalize=configs['normalize']).to(device)
        else:
            raise ValueError("Prompt method should be one of [None, expand, pad, fix, random]")
        # No need to compute gradient
        # network.requires_grad_(False)
        if args.prune_method == 'hydra':
            score_params= [param for param in network.parameters() if hasattr(param, 'is_score') and param.is_score]
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(list(visual_prompt.parameters())+score_params, lr=args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(list(visual_prompt.parameters())+score_params, lr=args.lr)
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(visual_prompt.parameters(), lr=args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=args.lr)
    else:
        visual_prompt = None
        network.requires_grad_(True)
        if args.prune_method == 'hydra':
            score_params= [param for param in network.parameters() if hasattr(param, 'is_score') and param.is_score]
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(list(network.parameters()) + score_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(list(network.parameters()) + score_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=0.1)
    
    return optimizer, scheduler, visual_prompt


def calculate_label_mapping(visual_prompt, network, train_loader, args):
    if args.prompt_method:
        if args.label_mapping_mode == 'rlm':
            print('Random Label Mapping')
            mapping_sequence = torch.randperm(1000)[:10]
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        elif args.label_mapping_mode in ('flm', 'ilm'):
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, network, train_loader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        else:
            mapping_sequence = None
            label_mapping = None
            raise ValueError("Not exist method ", args.label_mapping_mode)
    else:
        if args.label_mapping_mode == 'rlm':
            print('Random Label Mapping')
            mapping_sequence = torch.randperm(1000)[:10]
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        elif args.label_mapping_mode in ('flm', 'ilm'):
            mapping_sequence = generate_label_mapping_by_frequency_ordinary(network, train_loader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        else:
            mapping_sequence = None
            label_mapping = None
            raise ValueError("Not exist method ", args.label_mapping_mode)
        
    return label_mapping, mapping_sequence


def obtain_label_mapping(mapping_sequence):
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)

    return label_mapping