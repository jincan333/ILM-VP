import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import argparse
import time
from matplotlib import pyplot as plt
import copy
import warnings
import json

from utils import set_seed, setup_optimizer_and_prompt, calculate_label_mapping, obtain_label_mapping, save_args, get_masks
from get_model_dataset import choose_dataloader, get_model
from pruner import extract_mask, prune_model_custom, check_sparsity, remove_prune, pruning_model
from core import Masking, CosineDecay
from hydra import set_hydra_prune_rate, set_hydra_network


def main():    
    parser = argparse.ArgumentParser(description='PyTorch Visual Prompt + Prune Experiments')
    global args
    parser.add_argument('--prune_mode', type=str, default='vp_ff', choices=['vp_ff'], help='prune method implement ways')
    parser.add_argument('--second_phase', type=str, default='vp+ff_cotrain', choices=['freeze_vp+ff', 'vp+ff_cotrain', 'ff_then_vp'], help='actions after finding sub-network')
    parser.add_argument('--prune_method', type=str, default='hydra', choices=['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra'])
    parser.add_argument('--ckpt_directory', type=str, default='', help='sub-network ckpt directory')
    parser.add_argument('--ff_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--ff_scheduler', default='cosine', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--ff_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--ff_weight_decay', default=1e-4, type=float, help='finetune weight decay')
    parser.add_argument('--vp_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--vp_scheduler', default='multistep', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--vp_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--vp_weight_decay', default=1e-4, type=float, help='visual prompt weight decay')
    parser.add_argument('--hydra_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--hydra_scheduler', default='cosine', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--hydra_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--hydra_weight_decay', default=1e-4, type=float, help='hydra weight decay')
    parser.add_argument('--network', default='resnet18', choices=["resnet18", "resnet50", "vgg"])
    parser.add_argument('--dataset', default="cifar10", choices=['cifar10', 'cifar100', 'flowers102', 'dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--experiment_name', default='exp_new', type=str, help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', default=120, type=int, help='number of total eopchs to run')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--density_list', default='1,0.10,0.05,0.01', type=str, help='density list(1-sparsity), choose from 1,0.50,0.40,0.30,0.20,0.10,0.05')
    parser.add_argument('--label_mapping_mode', type=str, default='flm', choices=['flm', 'ilm'])

    ##################################### General setting ############################################
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='result', type=str)
    parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
    parser.add_argument('--resume_checkpoint', default='', help="resume checkpoint path")
    parser.add_argument('--print_freq', default=200, type=int, help='print frequency')

    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
    parser.add_argument('--randomcrop', type=int, default=0, help='dataset randomcrop.', choices=[0, 1])
    parser.add_argument('--input_size', type=int, default=224, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
    parser.add_argument('--pad_size', type=int, default=16, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[0, 16, 32, 48, 64, 80, 96, 112])
    parser.add_argument('--mask_size', type=int, default=183, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])

    ##################################### Architecture ############################################
    parser.add_argument('--label_mapping_interval', type=int, default=1, help='in ilm, the interval of epoch to implement label mapping')

    ##################################### Visual Prompt ############################################
    parser.add_argument('--output_size', type=int, default=224, help='image size after prompt, fix to 224')
    parser.add_argument('--prompt_method', type=str, default='pad', choices=['pad', 'fix', 'random', 'None'])

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--decreasing_step', default=[0.5,0.72], type = list, help='decreasing strategy')

    ##################################### Pruning setting #################################################
    parser.add_argument('--fix', default='true', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate used for dynamic sparse training (not used in this paper).')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
    parser.add_argument('--hydra_scaled_init', type=int, default=1, help='whether use scaled initialization for hydra or not.', choices=[0, 1])

    args = parser.parse_args()
    args.prompt_method=None if args.prompt_method=='None' else args.prompt_method        
    args.density_list=[float(i) for i in args.density_list.split(',')]
    if args.prune_method != 'hydra':
        args.ff_optimizer = 'sgd'
        args.ff_lr = 0.01
    print(json.dumps(vars(args), indent=4))
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    # Save Path
    save_path = os.path.join(args.save_dir, args.experiment_name, args.network, args.dataset, 
                'SECOND_PHASE'+args.second_phase, 'PRUNE_MODE'+args.prune_mode, 'PRUNE'+str(args.prune_method), 'VP'+str(args.prompt_method), 'LM'+str(args.label_mapping_mode),
                'SIZE'+str(args.output_size)+'_'+str(args.input_size)+'_'+str(args.pad_size)+'_'+str(args.mask_size),
                args.ff_optimizer+'_'+args.vp_optimizer+'_'+args.hydra_optimizer, 
                args.ff_scheduler+'_'+args.vp_scheduler+'_'+args.hydra_scheduler, 
                'LR'+str(args.ff_lr)+'_'+str(args.vp_lr)+'_'+str(args.hydra_lr),  
                'DENSITY'+str(args.density_list), 'EPOCHS'+str(args.epochs), 'SEED'+str(args.seed),'GPU'+str(args.gpu))
    os.makedirs(save_path, exist_ok=True)
    save_args(args, save_path+'/args.json')
    args.device=device
    print('Save path: ',save_path)
    # Network
    network = get_model(args)
    if args.prune_method == 'hydra':
        print('\nset hydra network.\n')
        network = set_hydra_network(network, args)
    print(network)
    # set phase
    print('*********************set phase as subnetwork**********************')
    phase = 'subnetwork'
    # DataLoader
    train_loader, val_loader, test_loader = choose_dataloader(args, phase)
    # Visual Prompt, Optimizer, and Scheduler
    visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler = setup_optimizer_and_prompt(network, args)
    # Label Mapping
    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
    print('mapping_sequence: ', mapping_sequence)
    # Prune initiate type
    # TODO need to add initialize from ckpt
    # init_ckpt = get_init_ckpt(args)
    # initalization = init_ckpt.state_dict() if init_ckpt else copy.deepcopy(network.state_dict())
    # network.load_state_dict(initalization)
    # visual_prompt = visual_prompt.load_state_dict(init_ckpt['visual_prompt']) if init_ckpt['visual_prompt'] else visual_prompt
    # mapping_sequence = init_ckpt['mapping_sequence']
    # label_mapping = obtain_label_mapping(mapping_sequence)
    # Initialize before train init dense network
    mask = None
    state_init = copy.deepcopy(network.state_dict())
    pre_state_init = copy.deepcopy(network.state_dict())
    visual_prompt_init = copy.deepcopy(visual_prompt.state_dict()) if visual_prompt else None
    visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler, checkpoint, best_acc, all_results = init_ckpt_vp_optimizer(
        network, visual_prompt_init, mapping_sequence, None, args)
    # Accuracy before prune
    test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
    print(f'Accuracy before Trian init dense network: {test_acc:.4f}')
    all_results['no_train_acc'] = test_acc
    # TODO need imp and omp need revise, maybe all need revise after we have ckpt
    # if args.prune_method in ('grasp', 'snip', 'synflow', 'hydra', 'vp_hydra'):
    #     start_state+=1 if start_state == 0 else start_state
    print(f'#######################Train init dense network for {args.prune_method}######################')
    print(f'#######################Train init dense network for {args.prune_method}######################')
    print(f'#######################Train init dense network for {args.prune_method}######################')
    for epoch in range(args.epochs):
        if args.prune_mode in ('no_tune', 'normal'):
            if args.prune_method in ('imp', 'random', 'omp'):
                train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                ff_optimizer=ff_optimizer, vp_optimizer=None, hydra_optimizer=None, 
                                ff_scheduler=ff_scheduler, vp_scheduler=None, hydra_scheduler=None)
            elif epoch==1:
                break
            else:    
                train_acc = 0
        elif args.prune_mode in ('vp', 'vp_ff'):
            if args.prune_method in ('imp', 'random', 'omp'):
                train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                ff_optimizer=ff_optimizer, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                                ff_scheduler=ff_scheduler, vp_scheduler=vp_scheduler, hydra_scheduler=None)
            elif args.prune_method in ('grasp', 'synflow', 'snip'):
                label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
                print('mapping_sequence: ', mapping_sequence)
                train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                ff_optimizer=None, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                                ff_scheduler=None, vp_scheduler=vp_scheduler, hydra_scheduler=None)
            elif epoch==1:
                break
            else:    
                train_acc = 0
        val_acc = evaluate(val_loader, network, label_mapping, visual_prompt)
        all_results['train_acc'].append(train_acc)
        all_results['val_acc'].append(val_acc)
        # Save CKPT
        checkpoint = {
            'state_dict': network.state_dict()
            ,'init_weight': state_init
            ,'masks': None
            ,"ff_optimizer": ff_optimizer.state_dict() if ff_optimizer else None
            ,'ff_scheduler': ff_scheduler.state_dict() if ff_scheduler else None
            ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
            ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
            ,"hydra_optimizer": hydra_optimizer.state_dict() if hydra_optimizer else None
            ,'hydra_scheduler': hydra_scheduler.state_dict() if hydra_scheduler else None
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence
            ,"val_best_acc": best_acc
            ,'ckpt_test_acc': 0
            ,'all_results': all_results
            ,"epoch": epoch
            ,'state': 0
        }
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint['val_best_acc'] = best_acc
            torch.save(checkpoint, os.path.join(save_path, '0best.pth'))
        # Plot training curve
        plot_train(all_results, save_path, 0)
    best_ckpt = torch.load(os.path.join(save_path, '0best.pth'))
    network.load_state_dict(best_ckpt['state_dict'])
    visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
    test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
    best_ckpt['ckpt_test_acc'] = test_acc
    torch.save(best_ckpt, os.path.join(save_path, '0best.pth'))
    print(f'Best CKPT Accuracy: {test_acc:.4f}')
    all_results['ckpt_test_acc'] = test_acc
    all_results['ckpt_epoch'] = best_ckpt['epoch']
    plot_train(all_results, save_path, 0)
    state_init = copy.deepcopy(best_ckpt['state_dict'])
    mapping_sequence_init = copy.deepcopy(best_ckpt['mapping_sequence'])
    visual_prompt_init = copy.deepcopy(visual_prompt.state_dict()) if visual_prompt else None
    for state in range(1, len(args.density_list)):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        # init
        phase = 'subnetwork'
        mask = None
        if args.prune_method == 'imp':
            pruning_model(network, (args.density_list[state-1] - args.density_list[state]) / args.density_list[state-1])
            current_mask = extract_mask(network.state_dict())
            remove_prune(network)
            network.load_state_dict(state_init)
        else:
            network.load_state_dict(state_init)
            if args.prune_method == 'hydra':
                if args.density_list[state] >= 0.1:
                    print('change ff optimizer to sgd')
                    args.ff_optimizer = 'sgd'
                    args.ff_lr = 0.01
                else:
                    print('change ff optimizer to adam')
                    args.ff_optimizer = 'adam'
                    args.ff_lr = 0.001
                set_hydra_prune_rate(network, 1)
        label_mapping = obtain_label_mapping(mapping_sequence_init)
        visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler, checkpoint, best_acc, all_results = init_ckpt_vp_optimizer(
            network, visual_prompt_init, mapping_sequence, None, args)
        test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Accuracy before prune: {test_acc:.4f}')
        # prune
        if args.prune_method == 'imp':
            print('IMP pruning')
            network.load_state_dict(pre_state_init)
            prune_model_custom(network, current_mask)
            check_sparsity(network)
        else:
            network, mask = prune_network(network, ff_optimizer, visual_prompt, label_mapping, train_loader, state, args)
            if args.prune_method in ('omp', 'random'):
                network.load_state_dict(pre_state_init)
                mask.apply_mask()
        masks = get_masks(mask) if mask else None
        if args.prune_method != 'hydra':
            label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
            print('mapping_sequence: ', mapping_sequence)
            test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
        elif args.prune_method == 'hydra':
            for epoch in range(args.epochs):
                if args.prune_mode in ('no_tune', 'normal'):
                    if args.label_mapping_mode == 'ilm':
                        label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
                        print('mapping_sequence: ', mapping_sequence)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=None, vp_optimizer=None, hydra_optimizer=hydra_optimizer, 
                                    ff_scheduler=None, vp_scheduler=None, hydra_scheduler=hydra_scheduler)
                elif args.prune_mode in ('vp', 'vp_ff'):
                    if args.label_mapping_mode == 'ilm':
                        label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
                        print('mapping_sequence: ', mapping_sequence)
                    # train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                    #                 ff_optimizer=None, vp_optimizer=vp_optimizer, hydra_optimizer=hydra_optimizer, 
                    #                 ff_scheduler=None, vp_scheduler=vp_scheduler, hydra_scheduler=hydra_scheduler)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=None, vp_optimizer=None, hydra_optimizer=hydra_optimizer, 
                                    ff_scheduler=None, vp_scheduler=None, hydra_scheduler=hydra_scheduler)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=None, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                                    ff_scheduler=None, vp_scheduler=vp_scheduler, hydra_scheduler=None)
                val_acc = evaluate(val_loader, network, label_mapping, visual_prompt)
                all_results['train_acc'].append(train_acc)
                all_results['val_acc'].append(val_acc)
                # Save CKPT
                checkpoint = {
                    'state_dict': network.state_dict()
                    ,'init_weight': state_init
                    ,'masks': None
                    ,"ff_optimizer": ff_optimizer.state_dict() if ff_optimizer else None
                    ,'ff_scheduler': ff_scheduler.state_dict() if ff_scheduler else None
                    ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
                    ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
                    ,"hydra_optimizer": hydra_optimizer.state_dict() if hydra_optimizer else None
                    ,'hydra_scheduler': hydra_scheduler.state_dict() if hydra_scheduler else None
                    ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
                    ,'mapping_sequence': mapping_sequence
                    ,"val_best_acc": best_acc
                    ,'ckpt_test_acc': 0
                    ,'all_results': all_results
                    ,"epoch": epoch
                    ,'state': 0
                }
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint['val_best_acc'] = best_acc
                    torch.save(checkpoint, os.path.join(save_path, str(state)+'after_prune.pth'))
                # Plot training curve
                plot_train(all_results, save_path, str(state)+'prune')
            best_ckpt = torch.load(os.path.join(save_path, str(state)+'after_prune.pth'))
            network.load_state_dict(best_ckpt['state_dict'])
            visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
            test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
            best_ckpt['ckpt_test_acc'] = test_acc
            torch.save(best_ckpt, os.path.join(save_path, str(state)+'after_prune.pth'))
            print(f'Best CKPT Accuracy: {test_acc:.4f}')
            all_results['ckpt_test_acc'] = test_acc
            all_results['ckpt_epoch'] = best_ckpt['epoch']
            plot_train(all_results, save_path, str(state)+'prune')
            # init
            visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler, checkpoint, best_acc, all_results = init_ckpt_vp_optimizer(
                network, visual_prompt.state_dict(), mapping_sequence, None, args)
        print(f'Accuracy after prune: {test_acc:.4f}')
        all_results['no_train_acc'] = test_acc
        torch.save(checkpoint, os.path.join(save_path, str(state)+'after_prune.pth'))
        # Second phase
        if args.prune_mode in ('normal', 'vp_ff'):
            print('*********************set phase as finetune**********************')
            phase = 'finetune'
            print('******************************************')
            print(f'pruning state {state} finetune')
            print('******************************************')
            for epoch in range(args.epochs):
                # 'freeze_vp+ff', 'vp+ff_cotrain', 'ff_then_vp'
                if args.second_phase in ('freeze_vp+ff', 'ff_then_vp'):
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=ff_optimizer, vp_optimizer=None, hydra_optimizer=None, 
                                    ff_scheduler=ff_scheduler, vp_scheduler=None, hydra_scheduler=None)
                elif args.second_phase == 'vp+ff_cotrain':
                    # train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                    #                 ff_optimizer=ff_optimizer, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                    #                 ff_scheduler=ff_scheduler, vp_scheduler=vp_scheduler, hydra_scheduler=None)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=ff_optimizer, vp_optimizer=None, hydra_optimizer=None, 
                                    ff_scheduler=ff_scheduler, vp_scheduler=None, hydra_scheduler=None)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=None, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                                    ff_scheduler=None, vp_scheduler=vp_scheduler, hydra_scheduler=None)

                val_acc = evaluate(val_loader, network, label_mapping, visual_prompt)
                all_results['train_acc'].append(train_acc)
                all_results['val_acc'].append(val_acc)
                # Save CKPT
                checkpoint = {
                    'state_dict': network.state_dict()
                    ,'init_weight': state_init
                    ,'mask': masks
                    ,"ff_optimizer": ff_optimizer.state_dict() if ff_optimizer else None
                    ,'ff_scheduler': ff_scheduler.state_dict() if ff_scheduler else None
                    ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
                    ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
                    ,"hydra_optimizer": hydra_optimizer.state_dict() if hydra_optimizer else None
                    ,'hydra_scheduler': hydra_scheduler.state_dict() if hydra_scheduler else None
                    ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
                    ,'mapping_sequence': mapping_sequence
                    ,"val_best_acc": best_acc
                    ,'ckpt_test_acc': 0
                    ,'all_results': all_results
                    ,"epoch": epoch
                    ,'state': 0
                }
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint['val_best_acc'] = best_acc
                    torch.save(checkpoint, os.path.join(save_path, str(state)+'best.pth'))
                # Plot training curve
                plot_train(all_results, save_path, state)
            best_ckpt = torch.load(os.path.join(save_path, str(state)+'best.pth'))
            network.load_state_dict(best_ckpt['state_dict'])
            visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
            test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
            best_ckpt['ckpt_test_acc'] = test_acc
            torch.save(best_ckpt, os.path.join(save_path, str(state)+'best.pth'))
            print(f'Best CKPT Accuracy: {test_acc:.4f}')
            all_results['ckpt_test_acc'] = test_acc
            all_results['ckpt_epoch'] = best_ckpt['epoch']
            plot_train(all_results, save_path, state)

            if args.second_phase == 'ff_then_vp':
                print('*********************start vp after finetune**********************')
                visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler, checkpoint, best_acc, all_results = init_ckpt_vp_optimizer(
                    network, visual_prompt.state_dict(), mapping_sequence, masks, args)
                for epoch in range(args.epochs):
                    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
                    print('mapping_sequence: ', mapping_sequence)
                    train_acc = train(train_loader, network, epoch, label_mapping, visual_prompt, mask, 
                                    ff_optimizer=None, vp_optimizer=vp_optimizer, hydra_optimizer=None, 
                                    ff_scheduler=None, vp_scheduler=vp_scheduler, hydra_scheduler=None)
                    val_acc = evaluate(val_loader, network, label_mapping, visual_prompt)
                    all_results['train_acc'].append(train_acc)
                    all_results['val_acc'].append(val_acc)
                    # Save CKPT
                    checkpoint = {
                        'state_dict': network.state_dict()
                        ,'init_weight': state_init
                        ,'mask': masks
                        ,"ff_optimizer": ff_optimizer.state_dict() if ff_optimizer else None
                        ,'ff_scheduler': ff_scheduler.state_dict() if ff_scheduler else None
                        ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
                        ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
                        ,"hydra_optimizer": hydra_optimizer.state_dict() if hydra_optimizer else None
                        ,'hydra_scheduler': hydra_scheduler.state_dict() if hydra_scheduler else None
                        ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
                        ,'mapping_sequence': mapping_sequence
                        ,"val_best_acc": best_acc
                        ,'ckpt_test_acc': 0
                        ,'all_results': all_results
                        ,"epoch": epoch
                        ,'state': 0
                    }
                    if val_acc > best_acc:
                        best_acc = val_acc
                        checkpoint['val_best_acc'] = best_acc
                        torch.save(checkpoint, os.path.join(save_path, str(state)+'best_vp.pth'))
                    # Plot training curve
                    plot_train(all_results, save_path, str(state)+'best_vp')
                best_ckpt = torch.load(os.path.join(save_path, str(state)+'best_vp.pth'))
                network.load_state_dict(best_ckpt['state_dict'])
                visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
                test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
                best_ckpt['ckpt_test_acc'] = test_acc
                torch.save(best_ckpt, os.path.join(save_path, str(state)+'best_vp.pth'))
                print(f'Best CKPT Accuracy: {test_acc:.4f}')
                all_results['ckpt_test_acc'] = test_acc
                all_results['ckpt_epoch'] = best_ckpt['epoch']
                plot_train(all_results, save_path, str(state)+'best_vp')


def train(train_loader, network, epoch, label_mapping, visual_prompt, mask, ff_optimizer, vp_optimizer, hydra_optimizer, ff_scheduler, vp_scheduler, hydra_scheduler):
    # switch to train mode
    if visual_prompt:
        visual_prompt.train()
    network.train()
    start = time.time()
    total_num = 0
    true_num = 0
    loss_sum = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        if ff_optimizer:
            ff_optimizer.zero_grad()
        if vp_optimizer:
            vp_optimizer.zero_grad()
        if hydra_optimizer:
            hydra_optimizer.zero_grad()
        if visual_prompt:
            fx = label_mapping(network(visual_prompt(x)))
        else:
            fx = label_mapping(network(x))
        loss = F.cross_entropy(fx, y, reduction='mean')
        loss.backward()
        if ff_optimizer:
            ff_optimizer.step()
        if vp_optimizer:
            vp_optimizer.step()
        if hydra_optimizer:
            hydra_optimizer.step()
        if mask:
            mask.apply_mask()
            mask.death_rate_decay.step()
            mask.death_rate = mask.death_rate_decay.get_dr()
            mask.steps += 1
            if mask.prune_every_k_steps is not None:
                if mask.steps % mask.prune_every_k_steps == 0:
                    mask.truncate_weights()
                    _, _ = mask.fired_masks_update()
                    mask.print_nonzero_counts()

    # scaler = GradScaler()
    # for i, (x, y) in enumerate(train_loader):
    #     x = x.cuda()
    #     y = y.cuda()
    #     if ff_optimizer:
    #         ff_optimizer.zero_grad()
    #     if vp_optimizer:
    #         vp_optimizer.zero_grad()
    #     if hydra_optimizer:
    #         hydra_optimizer.zero_grad()
    #     with autocast():
    #         if visual_prompt:
    #             fx = label_mapping(network(visual_prompt(x)))
    #         else:
    #             fx = label_mapping(network(x))
    #         loss = F.cross_entropy(fx, y, reduction='mean')
    #     scaler.scale(loss).backward()
    #     if ff_optimizer:
    #         scaler.step(ff_optimizer)
    #     if vp_optimizer:
    #         scaler.step(vp_optimizer)
    #     if hydra_optimizer:
    #         scaler.unscale_(hydra_optimizer)
    #         scaler.step(hydra_optimizer)
    #     if mask:
    #         mask.apply_mask()
    #         mask.death_rate_decay.step()
    #         mask.death_rate = mask.death_rate_decay.get_dr()
    #         mask.steps += 1
    #         if mask.prune_every_k_steps is not None:
    #             if mask.steps % mask.prune_every_k_steps == 0:
    #                 mask.truncate_weights()
    #                 _, _ = mask.fired_masks_update()
    #                 mask.print_nonzero_counts()
    #     scaler.update()

        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        train_acc= true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        # measure accuracy and record loss
        if (i+1) % args.print_freq == 0:
            end = time.time()
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'loss_sum {loss_sum:.4f}\t'
                f'Accuracy {train_acc:.4f}\t'
                f'Time {end-start:.2f}')
            start = time.time()
    end = time.time()
    print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
        f'loss_sum {loss_sum:.4f}\t'
        f'Accuracy {train_acc:.4f}\t'
        f'Time {end-start:.2f}')
    print(f'train_accuracy {train_acc:.3f}')
    if ff_scheduler:
        print('ff_lr: ', ff_optimizer.param_groups[0]['lr'])
        ff_scheduler.step()
    if vp_scheduler:
        print('vp_lr: ', vp_optimizer.param_groups[0]['lr'])
        vp_scheduler.step()
    if hydra_scheduler:
        print('hydra_lr: ', hydra_optimizer.param_groups[0]['lr'])
        hydra_scheduler.step()

    return train_acc


def evaluate(val_loader, network, label_mapping, visual_prompt):
    # switch to evaluate mode
    if visual_prompt:
        visual_prompt.eval()
    network.eval()
    total_num = 0
    true_num = 0
    loss_sum = 0
    for i, (x, y) in enumerate(val_loader):
        x = x.cuda()
        y = y.cuda()
        # compute output
        with torch.no_grad():
            if visual_prompt:
                fx = label_mapping(network(visual_prompt(x)))
            else:
                fx = label_mapping(network(x))
            loss = F.cross_entropy(fx, y, reduction='mean')
        # measure accuracy and record loss
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        if (i+1) % args.print_freq == 0:
            print(f'evaluate: [{i}/{len(val_loader)}]\t'
                f'Loss_sum {loss_sum:.4f}\t'
                f'Accuracy {test_acc:.4f}\t'
            )
    print(f'evaluate: [{i}/{len(val_loader)}]\t'
        f'Loss_sum {loss_sum:.4f}\t'
        f'Accuracy {test_acc:.4f}\t'
    )
    print(f'evaluate_accuracy {test_acc:.3f}')

    return test_acc


def prune_network(network, ff_optimizer, visual_prompt, label_mapping, train_loader, state, args):
    mask = None
    if args.prune_method in ('random', 'omp', 'grasp','snip','synflow'):
        print(f'{(args.prune_method).upper()} pruning')
        decay = CosineDecay(args.death_rate, len(train_loader)*args.epochs)
        mask = Masking(ff_optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                    redistribution_mode=args.redistribution, args=args, train_loader=train_loader, visual_prompt=visual_prompt, label_mapping=label_mapping)
        mask.add_module(network, density=args.density_list[state], sparse_init=args.prune_method)
    elif args.prune_method in ('hydra') and state > 0:
        print('Hydra Density Setting:', args.density_list[state])
        set_hydra_prune_rate(network, args.density_list[state])

    return network, mask


def init_ckpt_vp_optimizer(network, visual_prompt_init, mapping_sequence, masks, args):
    best_acc = 0.
    all_results={}
    all_results['train_acc'] = []
    all_results['val_acc'] = []
    visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler = setup_optimizer_and_prompt(network, args)
    visual_prompt.load_state_dict(visual_prompt_init) if visual_prompt_init else None
    checkpoint = {
            'state_dict': network.state_dict()
            ,'init_weight': None
            ,'mask': masks
            ,"ff_optimizer": ff_optimizer.state_dict() if ff_optimizer else None
            ,'ff_scheduler': ff_scheduler.state_dict() if ff_scheduler else None
            ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
            ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
            ,"hydra_optimizer": hydra_optimizer.state_dict() if hydra_optimizer else None
            ,'hydra_scheduler': hydra_scheduler.state_dict() if hydra_scheduler else None
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence
            ,"val_best_acc": 0
            ,'ckpt_test_acc': 0
            ,'all_results': all_results
            ,"epoch": 0
            ,'state': 0
        }

    return visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler, checkpoint, best_acc, all_results


def plot_train(all_results, save_path, state):
    if 'no_train_acc' in all_results:
        plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
    plt.plot(all_results['train_acc'], label='train_acc')
    plt.plot(all_results['val_acc'], label='val_acc')
    if 'ckpt_test_acc' in all_results:
        plt.scatter(all_results['ckpt_epoch'], all_results['ckpt_test_acc'], label='ckpt_test_acc', color='red', marker='s')
    plt.legend()
    plt.title(save_path, fontsize = 'xx-small')
    plt.savefig(os.path.join(save_path, str(state)+'train.png'))
    plt.close()


if __name__ == '__main__':
    main()
    