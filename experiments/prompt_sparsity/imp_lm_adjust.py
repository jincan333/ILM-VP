import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import argparse
import time
from matplotlib import pyplot as plt
import copy

import sys
sys.path.append(".")
from cfg import *
from parser import add_args
from utils import setup_model_dataset, set_seed
from algorithms import label_mapping_base, generate_label_mapping_by_frequency_ordinary
from pruner import extract_mask, prune_model_custom, check_sparsity, pruning_model_random, remove_prune, pruning_model


def main():
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch IMP Experiments')
    add_args(parser)
    global args
    args = parser.parse_args()
    print(args)
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    # Save Path
    save_path = os.path.join(args.save_dir, args.experiment_name, str(args.sparse_init), str(args.label_mapping_mode), f"gpu{args.gpu}")
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    print('Save path: ',save_path)
    # Model and Dataset
    network, train_loader, val_loader, test_loader = setup_model_dataset(args)
    # network.fc = torch.nn.Linear(512, 10).to(device)
    network.cuda()
    network.requires_grad_(True)
    print(network)
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decreasing_step, gamma=0.1)
    # Label Mapping
    if args.label_mapping_mode == 'rlm':
        print('Random Label Mapping')
        mapping_sequence = torch.randperm(1000)[:10]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    elif args.label_mapping_mode == 'flm':
        mapping_sequence = generate_label_mapping_by_frequency_ordinary(network, train_loader, device=device)
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    elif args.label_mapping_mode == None:
        label_mapping = None
    # IMP prune initiate type
    # TODO what's the meaning of these?
    if args.imp_prune_type == 'lt':
        print('lottery tickets setting (rewind to the same random init)')
        initalization = copy.deepcopy(network.state_dict())
    elif args.imp_prune_type == 'pt':
        print('lottery tickets from best dense weight')
        initalization = None
    elif args.imp_prune_type == 'rewind_lt':
        print('lottery tickets with early weight rewinding')
        initalization = None
    else:
        raise ValueError('unknown imp_prune_type')
    # Resume from checkpoint
    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state>0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(network, current_mask)
            check_sparsity(network)
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # Scheduler
            if args.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.lr_scheduler == 'multistep':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decreasing_step, gamma=0.1)
        network.load_state_dict(checkpoint['state_dict'])
        # adding an extra forward process to enable the masks
        x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
        network.eval()
        with torch.no_grad:
            network(x_rand)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_acc=', best_acc)
    else:
        best_acc = 0. 
        start_epoch = 0
        start_state = 0
        all_results={}
        all_results['train_acc'] = []
        all_results['val_acc'] = []
    
    print('######################################## Start Standard Training Iterative Pruning ########################################')
    for state in range(start_state, args.imp_pruning_times):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        # Accuracy without train
        test_acc, test_loss = evaluate(test_loader, network, label_mapping)
        print(f'Accuracy without train: {test_acc:.4f}, loss {test_loss:.4f}')
        all_results['no_train_acc'] = test_acc
        for epoch in range(args.epochs):
            print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
            # Train
            if args.label_mapping_mode == 'ilm':
                mapping_sequence = generate_label_mapping_by_frequency_ordinary(network, train_loader, device=device)
                label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
            train_acc, train_loss = train(train_loader, network, optimizer, epoch, label_mapping)
            scheduler.step()
            logger.add_scalar("train/acc", train_acc, epoch)
            logger.add_scalar("train/loss", train_loss, epoch)
            # Save rewind state
            if state ==0:
                if(epoch+1 == args.imp_rewind_epoch):
                    torch.save(network.state_dict(), os.path.join(save_path, f'epoch_{epoch+1}_rewind_weight.pth'))
                    if args.imp_prune_type == 'rewind_lt':
                        initalization = copy.deepcopy(network.state_dict())

            # Validate
            val_acc, val_loss = evaluate(val_loader, network, label_mapping)
            logger.add_scalar("validate/acc", val_acc, epoch)
            logger.add_scalar("validate/loss", val_loss, epoch)

            all_results['train_acc'].append(train_acc)
            all_results['val_acc'].append(val_acc)
            # Save CKPT
            checkpoint = {
                'state_dict': network.state_dict()
                ,"optimizer_dict": optimizer.state_dict()
                ,'scheduler': scheduler.state_dict()
                ,"epoch": epoch
                ,"val_best_acc": best_acc
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
            plt.savefig(os.path.join(save_path, str(state)+'train.png'))
            plt.close()
        # Test 
        best_ckpt = torch.load(os.path.join(save_path, str(state)+'best.pth'))
        network.load_state_dict(best_ckpt['state_dict'])
        test_acc, test_loss = evaluate(test_loader, network, label_mapping)
        best_ckpt['ckpt_test_acc'] = test_acc
        torch.save(best_ckpt, os.path.join(save_path, str(state)+'best.pth'))
        print(f'Best CKPT Accuracy: {test_acc:.4f}, loss {test_loss:.4f}')
        all_results['ckpt_test_acc'] = test_acc
        all_results['ckpt_epoch'] = best_ckpt['epoch']
        plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
        plt.plot(all_results['train_acc'], label='train_acc')
        plt.plot(all_results['val_acc'], label='val_acc')
        plt.scatter(all_results['ckpt_epoch'], all_results['ckpt_test_acc'], label='skpt_test_acc', color='red', marker='s')
        plt.legend()
        plt.savefig(os.path.join(save_path, str(state)+'train.png'))
        plt.close()

        # Initialize 
        best_acc = 0. 
        start_epoch = 0
        start_state = 0
        all_results={}
        all_results['train_acc'] = []
        all_results['val_acc'] = []
        if args.imp_prune_type == 'pt':
            print('* loading pretrained weight')
            initalization = torch.load(os.path.join(save_path, '0best.pth'))['state_dict']
        # Prune
        if args.imp_random_prune:
            print('random pruning')
            pruning_model_random(network, 1-args.imp_density)
        else:
            print('L1 pruning')
            pruning_model(network, 1-args.imp_density)
        remain_weight = check_sparsity(network)
        current_mask = extract_mask(network.state_dict())
        remove_prune(network)
        # Rewind
        network.load_state_dict(initalization)
        prune_model_custom(network, current_mask)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.lr_scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decreasing_step, gamma=0.1)
        if args.imp_prune_type == 'rewind_lt':
            if args.imp_rewind_epoch:
                # learning rate rewinding 
                for _ in range(args.imp_rewind_epoch):
                    scheduler.step()
            start_epoch = args.imp_rewind_epoch


def train(train_loader, model, optimizer, epoch, label_mapping):
    start = time.time()
    total_num = 0
    true_num = 0
    loss_sum = 0
    scaler = GradScaler()
    # switch to train mode
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        with autocast():
            if label_mapping:
                fx = label_mapping(model(x))
            else:
                fx = model(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        scaler.scale(loss).backward()
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
                f'Loss {loss_sum:.4f}\t'
                f'Accuracy {train_acc:.4f}\t'
                f'Time {end-start:.2f}')
            start = time.time()
    print(f'train_accuracy {train_acc:.3f}')

    return train_acc, loss_sum


def evaluate(val_loader, model, label_mapping):
    # switch to evaluate mode
    model.eval()
    total_num = 0
    true_num = 0
    loss_sum = 0
    for i, (x, y) in enumerate(val_loader):
        x = x.cuda()
        y = y.cuda()
        # compute output
        with torch.no_grad():
            if label_mapping:
                fx = label_mapping(model(x))
            else:
                fx = model(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        # measure accuracy and record loss
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        if i % args.print_freq == 0:
            print(f'evaluate: [{i}/{len(val_loader)}]\t'
                f'Loss {loss_sum:.4f}\t'
                f'Accuracy {test_acc:.4f}\t'
            )
    print(f'evaluate_accuracy {test_acc:.3f}')

    return test_acc, loss_sum


if __name__ == '__main__':
    main()
    