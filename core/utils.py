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


def get_optimizer(parameters, optimizer, scheduler, lr, weight_decay, args):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('optimizer should be one of [sgd, adam]')

    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=0.1)
    else:
        raise ValueError('scheduler should be one of [cosine, multistep]')

    return optimizer, scheduler


def setup_optimizer_and_prompt(network, args):
    device = args.device
    normalize = args.normalize
    visual_prompt = None
    hydra_optimizer, hydra_scheduler = None, None
    vp_optimizer, vp_scheduler = None, None
    ff_optimizer, ff_scheduler = None, None
    ff_params = network.parameters()

    if args.prune_mode in ('vp', 'vp_ff'):
        if args.prompt_method == 'pad':
            visual_prompt = PadVisualPrompt(args, normalize=normalize).to(device)
        elif args.prompt_method == 'fix':
            visual_prompt = FixVisualPrompt(args, normalize=normalize).to(device)
        elif args.prompt_method == 'random':
            visual_prompt = RandomVisualPrompt(args, normalize=normalize).to(device)
        else:
            raise ValueError("Prompt method should be one of [pad, fix, random]")
        vp_optimizer, vp_scheduler = get_optimizer(visual_prompt.parameters(), args.vp_optimizer, args.vp_scheduler, args.vp_lr, args.vp_weight_decay, args)

    if args.prune_method == 'hydra':
        if args.prune_mode == 'vp_ff':
            score_params = network.parameters()
        elif args.prune_mode == 'normal':
            score_params = [param for param in network.parameters() if hasattr(param, 'is_score') and param.is_score]
        else:
            raise ValueError("Prune Mode should be one of normal vp_ff")
        hydra_optimizer, hydra_scheduler = get_optimizer(score_params, args.hydra_optimizer, args.hydra_scheduler, args.hydra_lr, args.hydra_weight_decay, args)
        ff_params = [param for param in network.parameters() if not hasattr(param, 'is_score')]
    
    ff_optimizer, ff_scheduler = get_optimizer(ff_params, args.ff_optimizer, args.ff_scheduler, args.ff_lr, args.ff_weight_decay, args)
    
    return visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler


def calculate_label_mapping(visual_prompt, network, train_loader, args):
    if visual_prompt:
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
    with open(file_path, 'w') as file:
        json.dump(vars(args), file)


def load_args(file_path):
    with open(file_path, 'r') as file:
        load_args = json.load(file)

    return load_args


def get_init_ckpt(args):
    ckpt = None
    if args.prune_mode == 'normal' and args.prune_method in ('random', 'imp', 'omp'):
        ckpt = torch.load('ckpts/0best.pth')

    return ckpt


def get_masks(mask):
    masks = {}
    for module in mask.modules:
        for name, tensor in module.named_parameters():
            if name in mask.masks:
                masks[name] = mask.masks[name]
    
    return masks