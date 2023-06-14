'''
    setup model and datasets
'''
import torch
import numpy as np
import random
from functools import partial
import warnings
import pickle
import json

from visual_prompt import ExpansiveVisualPrompt, PadVisualPrompt, FixVisualPrompt, RandomVisualPrompt
from label_mapping import label_mapping_base, generate_label_mapping_by_frequency, generate_label_mapping_by_frequency_ordinary
__all__ = [
    'set_seed'
    ,'setup_optimizer_and_prompt'
    ,'calculate_label_mapping'
    ,'obtain_label_mapping'
    ,'save_args'
    ,'load_args'
]


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_optimizer_and_prompt(network, args):
    device = args.device
    normalize = args.normalize
    if args.prompt_method:
        # get prompt parameter
        if args.prompt_method == 'expand':
            visual_prompt = ExpansiveVisualPrompt(args, normalize=normalize).to(device)
        elif args.prompt_method == 'pad':
            visual_prompt = PadVisualPrompt(args, normalize=normalize).to(device)
        elif args.prompt_method == 'fix':
            visual_prompt = FixVisualPrompt(args, normalize=normalize).to(device)
        elif args.prompt_method == 'random':
            visual_prompt = RandomVisualPrompt(args, normalize=normalize).to(device)
        else:
            raise ValueError("Prompt method should be one of [None, expand, pad, fix, random]")
        # set optimizer
        # whether get network parameters
        if args.is_finetune:
            if args.prune_method == 'hydra':
                score_params= [param for param in network.parameters() if hasattr(param, 'is_score') and param.is_score]
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(list(network.parameters())+list(visual_prompt.parameters())+score_params, lr=args.lr, momentum=args.momentum)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(list(network.parameters())+list(visual_prompt.parameters())+score_params, lr=args.lr)
            else:
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(list(network.parameters())+list(visual_prompt.parameters()), lr=args.lr, momentum=args.momentum)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(list(network.parameters())+list(visual_prompt.parameters()), lr=args.lr)
        else:
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
        # set optimizer
        # whether get network parameters
        if args.is_finetune:
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
        else:
            if args.prune_method == 'hydra':
                score_params= [param for param in network.parameters() if hasattr(param, 'is_score') and param.is_score]
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(score_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(score_params, lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = None
    if args.lr_scheduler == 'cosine' and optimizer:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'multistep' and optimizer:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=0.1)
    else:
        scheduler = None

    return optimizer, scheduler, visual_prompt


def calculate_label_mapping(visual_prompt, network, train_loader, args):
    if args.prompt_method:
        if args.label_mapping_mode == 'rlm':
            print('Random Label Mapping')
            mapping_sequence = torch.randperm(1000)[:args.class_cnt]
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        elif args.label_mapping_mode in ('flm', 'ilm'):
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, network, train_loader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        else:
            mapping_sequence = None
            label_mapping = None
            warnings.warn('No Label Mapping!')

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
            warnings.warn('No Label Mapping!')
        
    return label_mapping, mapping_sequence


def obtain_label_mapping(mapping_sequence):
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)

    return label_mapping


def save_args(args, file_path):
    with open(file_path, 'wb') as file:
        json.dump(vars(args), file)


def load_args(file_path):
    with open(file_path, 'rb') as file:
        load_args = json.load(file)

    return load_args