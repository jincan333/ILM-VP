import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import argparse
import time
from matplotlib import pyplot as plt

import sys
sys.path.append(".")
from cfg import *
from parser import add_args
from utils import setup_model_dataset, set_seed
from algorithms import label_mapping_base, generate_label_mapping_by_frequency_ordinary

def main():
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch Full_Finetune Experiments')
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
    # Make Dir
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    # Accuracy without train
    test_acc, test_loss = evaluate(test_loader, network, label_mapping)
    print(f'Accuracy without train: {test_acc:.4f}, loss {test_loss:.4f}')

    best_acc = 0. 
    all_results={}
    all_results['no_train_acc'] = test_acc
    all_results['train_acc'] = []
    all_results['val_acc'] = []
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
            ,"epoch": epoch
            ,"best_acc": best_acc
        }
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint['best_acc'] = best_acc
            torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
        torch.save(checkpoint, os.path.join(save_path, 'ckpt.pth'))
        # Plot training curve
        plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
        plt.plot(all_results['train_acc'], label='train_acc')
        plt.plot(all_results['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'train.png'))
        plt.close()
    # Test 
    best_ckpt = torch.load(os.path.join(save_path, 'best.pth'))
    network.load_state_dict(best_ckpt['state_dict'])
    test_acc, test_loss = evaluate(test_loader, network, label_mapping)
    print(f'Best CKPT Accuracy: {test_acc:.4f}, loss {test_loss:.4f}')
    all_results['ckpt_test_acc'] = test_acc
    all_results['ckpt_epoch'] = best_ckpt['epoch']
    plt.scatter(0, all_results['no_train_acc'], label='raw_acc', color='green', marker='o')
    plt.plot(all_results['train_acc'], label='train_acc')
    plt.plot(all_results['val_acc'], label='val_acc')
    plt.scatter(all_results['ckpt_epoch'], all_results['ckpt_test_acc'], label='skpt_test_acc', color='red', marker='s')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'train.png'))
    plt.close()
    
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
    