import os
import torch
from torch.nn import functional as F
from torch.nn import Conv2d, Linear
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from matplotlib import pyplot as plt
import copy

import sys
sys.path.append(".")
from cfg import *
from parser_func import add_args, save_args
from utils import setup_model_dataset, set_seed, setup_optimizer_scheduler, calculate_label_mapping, obtain_label_mapping

from pruner import extract_mask, prune_model_custom, check_sparsity, pruning_model_random, remove_prune, pruning_model
from core import Masking, CosineDecay
from hydra import get_layers, replace_layers, setPruneRate, initialize_scaled_score

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Visual Prompt + Prune + Label Mapping Experiments')
    add_args(parser)
    global args
    args = parser.parse_args()
    print(args)
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    # Save Path
    save_path = os.path.join(args.save_dir, args.network, args.dataset, 'VP'+str(args.prompt_method), 'PRUNE'+str(args.prune_method), 'LM'+str(args.label_mapping_mode), 
                'LP'+str(args.is_adjust_linear_head), args.experiment_name, args.optimizer, 'LR'+str(args.lr), args.lr_scheduler, 'EPOCHS'+str(args.epochs), 'IMAGESIZE'+str(args.input_size)+'_'+str(args.pad_size)+'_'+str(args.mask_size), 
                'GPU'+str(args.gpu), 'DENSITY'+str(args.k))
    os.makedirs(save_path, exist_ok=True)
    save_args(args, save_path+'args.pkl')
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    print('Save path: ',save_path)
    # Network and Dataset
    network, train_loader, val_loader, test_loader, configs = setup_model_dataset(args)
    if args.is_adjust_linear_head:
        network.fc = torch.nn.Linear(512, 10).to(device)
    if args.prune_method == 'hydra':
        cl, ll = get_layers(args.layer_type)
        network=replace_layers(network,Conv2d,cl,args.ChannelPrune)
        network=replace_layers(network,Linear,ll,args.ChannelPrune)
        network.to(device)
        print(network)
        # initialize_scaled_score(network)
    # Optimizer and Scheduler
    optimizer, scheduler, visual_prompt = setup_optimizer_scheduler(network, configs, device, args)
    # Label Mapping
    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
    # Prune initiate type
    initalization = copy.deepcopy(network.state_dict())
    best_acc = 0. 
    start_state = 0
    all_results={}
    all_results['train_acc'] = []
    all_results['val_acc'] = []
    # Accuracy without train
    test_acc, test_loss = evaluate(test_loader, network, label_mapping, visual_prompt)
    print(f'Accuracy without train: {test_acc:.4f}, loss {test_loss:.4f}')
    all_results['no_train_acc'] = test_acc
    checkpoint = {
            'state_dict': network.state_dict()
            ,"optimizer_dict": optimizer.state_dict()
            ,'scheduler': scheduler.state_dict()
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence if args.label_mapping_mode else None
            ,"epoch": 0
            ,"val_best_acc": 0
            ,'ckpt_test_acc': test_acc
            ,'all_results': all_results
            ,'init_weight': initalization
            ,'state': start_state
        }
    torch.save(checkpoint, os.path.join(save_path, str(start_state)+'best.pth'))
    if args.prune_method in ('omp', 'grasp', 'hydra'):
        start_state+=1
    print('######################################## Start Standard Training Iterative Pruning ########################################')
    for state in range(start_state, args.pruning_times):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        mask = None
        for epoch in range(args.epochs * args.multiplier):
            print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch == args.warmup:
            # OMP, Grasp Prune
                if args.prune_method in ('omp', 'grasp') and state > 0:
                    decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
                    mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                                redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
                    mask.add_module(network, label_mapping=label_mapping, sparse_init=args.prune_method, density=(args.density)**state)
                elif args.prune_method == 'hydra' and state > 0:
                    print('Hydra Density Setting:', (args.density)**state)
                    setPruneRate(network, cl, ll, (args.density)**state)
                else:
                    pass
            # Train
            if args.label_mapping_mode == 'ilm' and epoch % args.label_mapping_interval == 0:
                label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
            train_acc, train_loss = train(train_loader, network, optimizer, epoch, label_mapping, visual_prompt, mask)
            scheduler.step()
            logger.add_scalar("train/acc", train_acc, epoch)
            logger.add_scalar("train/loss", train_loss, epoch)

            # Validate
            val_acc, val_loss = evaluate(val_loader, network, label_mapping, visual_prompt)
            logger.add_scalar("validate/acc", val_acc, epoch)
            logger.add_scalar("validate/loss", val_loss, epoch)

            all_results['train_acc'].append(train_acc)
            all_results['val_acc'].append(val_acc)
            # Save CKPT
            checkpoint = {
                'state_dict': network.state_dict()
                ,"optimizer_dict": optimizer.state_dict()
                ,'scheduler': scheduler.state_dict()
                ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
                ,'mapping_sequence': mapping_sequence if args.label_mapping_mode else None
                ,"epoch": epoch
                ,"val_best_acc": best_acc
                ,'ckpt_test_acc': 0
                ,'all_results': all_results
                ,'init_weight': initalization
                ,'state': state
            }
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint['val_best_acc'] = best_acc
                torch.save(checkpoint, os.path.join(save_path, str(state)+'best.pth'))
            torch.save(checkpoint, os.path.join(save_path, str(state)+'ckpt.pth'))
            # Plot training curve
            plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
            plt.plot(all_results['train_acc'], label='train_acc')
            plt.plot(all_results['val_acc'], label='val_acc')
            plt.legend()
            plt.title(save_path, fontsize = 'xx-small')
            plt.savefig(os.path.join(save_path, str(state)+'train.png'))
            plt.close()
        # Test 
        best_ckpt = torch.load(os.path.join(save_path, str(state)+'best.pth'))
        network.load_state_dict(best_ckpt['state_dict'])
        if args.label_mapping_mode:
            mapping_sequence = best_ckpt['mapping_sequence']
            label_mapping = obtain_label_mapping(mapping_sequence)
        visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
        test_acc, test_loss = evaluate(test_loader, network, label_mapping, visual_prompt)
        best_ckpt['ckpt_test_acc'] = test_acc
        torch.save(best_ckpt, os.path.join(save_path, str(state)+'best.pth'))
        print(f'Best CKPT Accuracy: {test_acc:.4f}, loss {test_loss:.4f}')
        all_results['ckpt_test_acc'] = test_acc
        all_results['ckpt_epoch'] = best_ckpt['epoch']
        plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
        plt.plot(all_results['train_acc'], label='train_acc')
        plt.plot(all_results['val_acc'], label='val_acc')
        plt.scatter(all_results['ckpt_epoch'], all_results['ckpt_test_acc'], label='ckpt_test_acc', color='red', marker='s')
        plt.legend()
        plt.title(save_path, fontsize = 'xx-small')
        plt.savefig(os.path.join(save_path, str(state)+'train.png'))
        plt.close()

        # Initialize 
        best_acc = 0. 
        all_results={}
        all_results['train_acc'] = []
        all_results['val_acc'] = []
        if (args.prune_method == 'imp' and args.imp_prune_type == 'pt') or args.prune_method in ('omp', 'grasp'):
            print('* loading pretrained weight')
            ckpt0 = torch.load(os.path.join(save_path, '0best.pth'))
            initalization = ckpt0['state_dict']
            mapping_sequence = ckpt0['mapping_sequence']
            label_mapping = obtain_label_mapping(mapping_sequence)
        # Network Initialize
        if args.prune_method == 'imp':
            print('L1 pruning')
            pruning_model(network, 1-args.density)
            current_mask = extract_mask(network.state_dict())
            remove_prune(network)
            network.load_state_dict(initalization)
            prune_model_custom(network, current_mask)
            check_sparsity(network)
        else:
            network.load_state_dict(initalization)
        # Optimizer, Schedule and Visual Prompt Initialize
        optimizer, scheduler, visual_prompt = setup_optimizer_scheduler(network, configs, device, args)
        # Save for the next iteration
        test_acc, test_loss = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Accuracy without train: {test_acc:.4f}, loss {test_loss:.4f}')
        all_results['no_train_acc'] = test_acc
        checkpoint = {
            'state_dict': network.state_dict()
            ,"optimizer_dict": optimizer.state_dict()
            ,'scheduler': scheduler.state_dict()
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence if args.label_mapping_mode else None
            ,"epoch": 0
            ,"val_best_acc": 0
            ,'ckpt_test_acc': test_acc
            ,'all_results': all_results
            ,'init_weight': initalization
            ,'state': state + 1
        }
        torch.save(checkpoint, os.path.join(save_path, str(state+1)+'best.pth'))


def train(train_loader, network, optimizer, epoch, label_mapping, visual_prompt, mask):
    # switch to train mode
    if visual_prompt:
        visual_prompt.train()
        network.train()
    else:
        network.train()
    start = time.time()
    total_num = 0
    true_num = 0
    loss_sum = 0
    scaler = GradScaler()
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        with autocast():
            if visual_prompt:
                if label_mapping:
                    fx = label_mapping(network(visual_prompt(x)))
                else:
                    fx = network(visual_prompt(x))
            elif label_mapping:
                fx = label_mapping(network(x))
            else:
                fx = network(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        scaler.scale(loss).backward()
        if mask:
            scaler.step(mask.optimizer)
            mask.apply_mask()
            mask.death_rate_decay.step()
            mask.death_rate = mask.death_rate_decay.get_dr()
            mask.steps += 1
            if mask.prune_every_k_steps is not None:
                if mask.steps % mask.prune_every_k_steps == 0:
                    mask.truncate_weights()
                    _, _ = mask.fired_masks_update()
                    mask.print_nonzero_counts()
        else:
            scaler.step(optimizer)
        scaler.update()

        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        train_acc= true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        # measure accuracy and record loss
        if i % args.print_freq == 0:
            end = time.time()
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'loss_sum {loss_sum:.4f}\t'
                f'Accuracy {train_acc:.4f}\t'
                f'Time {end-start:.2f}')
            start = time.time()
    print(f'train_accuracy {train_acc:.3f}')

    return train_acc, loss_sum


def evaluate(val_loader, network, label_mapping, visual_prompt):
    # switch to evaluate mode
    if visual_prompt:
        visual_prompt.eval()
        network.eval()
    else:
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
                if label_mapping:
                    fx = label_mapping(network(visual_prompt(x)))
                else:
                    fx = network(visual_prompt(x))
            elif label_mapping:
                fx = label_mapping(network(x))
            else:
                fx = network(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        # measure accuracy and record loss
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        if i % args.print_freq == 0:
            print(f'evaluate: [{i}/{len(val_loader)}]\t'
                f'Loss_sum {loss_sum:.4f}\t'
                f'Accuracy {test_acc:.4f}\t'
            )
    print(f'evaluate_accuracy {test_acc:.3f}')

    return test_acc, loss_sum


if __name__ == '__main__':
    main()
    