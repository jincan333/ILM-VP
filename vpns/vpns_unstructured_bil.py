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
from unstructured_network_with_score import set_prune_threshold, set_scored_network, switch_to_finetune, switch_to_bilevel, switch_to_prune


def main():    
    parser = argparse.ArgumentParser(description='PyTorch Visual Prompt + Prune Experiments')
    global args
    parser.add_argument('--prune_mode', type=str, default='score+vp_weight', choices=['score+vp_weight', 'score+weight_vp', 'weight+vp_score', 'score+vp_weight+vp', 'score_weight'], help='prune method implement ways')
    parser.add_argument('--prune_method', type=str, default='vpns', choices=['vpns', 'bip'])
    parser.add_argument('--ckpt_directory', type=str, default='', help='sub-network ckpt directory')
    parser.add_argument('--weight_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--weight_scheduler', default='cosine', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--weight_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight_weight_decay', default=1e-4, type=float, help='finetune weight decay')
    parser.add_argument('--vp_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--vp_scheduler', default='multistep', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--vp_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--vp_weight_decay', default=1e-4, type=float, help='visual prompt weight decay')
    parser.add_argument('--score_optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--score_scheduler', default='cosine', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--score_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--score_weight_decay', default=1e-4, type=float, help='hydra weight decay')
    parser.add_argument('--network', default='resnet18', choices=["resnet18", "resnet50", "vgg"])
    parser.add_argument('--dataset', default="cifar100", choices=['cifar10', 'cifar100', 'flowers102', 'dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--experiment_name', default='exp_new', type=str, help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', default=80, type=int, help='number of total eopchs to run')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--density_list', default='1,0.50,0.20,0.10', type=str, help='density list(1-sparsity), choose from 1,0.50,0.40,0.30,0.20,0.10,0.05')
    parser.add_argument('--score_vp_ratio', type=float, default=5, choices=[20,10,9,8,7,6,5,4,3,2,1])
    parser.add_argument('--label_mapping_mode', type=str, default='flm', choices=['flm', 'ilm'])
    parser.add_argument('--bil_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--global_vp_data', default=1, type=int, choices=[0,1], required=True, help='while using visual prompt, whether use vp_data in all training phase')

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

    ##################################### Visual Prompt ############################################
    parser.add_argument('--output_size', type=int, default=224, help='image size after prompt, fix to 224')
    parser.add_argument('--prompt_method', type=str, default='pad', choices=['pad', 'fix', 'random', 'None'])

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--decreasing_step', default=[0.5,0.72], type = list, help='decreasing strategy')

    ##################################### Pruning setting #################################################
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    parser.add_argument('--hydra_scaled_init', type=int, default=1, help='whether use scaled initialization for hydra or not.', choices=[0, 1])

    args = parser.parse_args()
    args.prompt_method=None if args.prompt_method=='None' else args.prompt_method        
    args.density_list=[float(i) for i in args.density_list.split(',')]
    args.step_division=int(args.score_vp_ratio)+1 if args.score_vp_ratio>=1 else int(1/args.score_vp_ratio)+1
    args.current_steps=0
    print(json.dumps(vars(args), indent=4))
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    # Save Path
    save_path = os.path.join(args.save_dir, args.experiment_name, args.network, args.dataset, 
                'PRUNE_MODE'+args.prune_mode, 'PRUNE'+str(args.prune_method), 'VP'+str(args.prompt_method),
                'SIZE'+str(args.output_size)+'_'+str(args.input_size)+'_'+str(args.pad_size)+'_'+str(args.mask_size),
                args.weight_optimizer+'_'+args.vp_optimizer+'_'+args.score_optimizer, 
                args.weight_scheduler+'_'+args.vp_scheduler+'_'+args.score_scheduler, 
                'LR'+str(args.weight_lr)+'_'+str(args.vp_lr)+'_'+str(args.score_lr),  
                'GLOBAL_DATA'+str(args.global_vp_data)+'DENSITY'+str(args.density_list), 'EPOCHS'+str(args.epochs), 'SEED'+str(args.seed),'GPU'+str(args.gpu))
    os.makedirs(save_path, exist_ok=True)
    save_args(args, save_path+'/args.json')
    args.device=device
    print('Save path: ',save_path)
    # Network
    network = get_model(args)
    network = set_scored_network(network, args)
    print(network)
    # DataLoader
    train_loader, val_loader, test_loader = choose_dataloader(args)
    # Visual Prompt, Optimizer, and Scheduler
    visual_prompt, score_optimizer, score_scheduler, vp_optimizer, vp_scheduler, weight_optimizer, weight_scheduler = setup_optimizer_and_prompt(network, args)
    # Label Mapping
    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
    print('mapping_sequence: ', mapping_sequence)
    # initiate
    mask = None
    pre_state_init = copy.deepcopy(network.state_dict())
    visual_prompt_init = copy.deepcopy(visual_prompt.state_dict()) if visual_prompt else None
    mapping_sequence_init = mapping_sequence
    visual_prompt, score_optimizer, score_scheduler, vp_optimizer, vp_scheduler, weight_optimizer, weight_scheduler, checkpoint, best_acc, all_results = init_ckpt_vp_optimizer(
        network, visual_prompt_init, mapping_sequence, None, args)
    
    print(f'####################### Prune and Train network for {args.prune_method} ######################') 
    for state in range(1, len(args.density_list)):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        # init
        mask = None
        network.load_state_dict(pre_state_init)
        if args.prune_method in ('vpns', 'hydra', 'bip'):
            set_prune_threshold(network, 1)
        label_mapping = obtain_label_mapping(mapping_sequence_init)
        visual_prompt, score_optimizer, score_scheduler, vp_optimizer, vp_scheduler, weight_optimizer, weight_scheduler, checkpoint, \
            best_acc, all_results = init_ckpt_vp_optimizer(network, visual_prompt_init, mapping_sequence, None, args)
        test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Accuracy before prune: {test_acc:.4f}')
        # prune
        network, mask = prune_network(network, weight_optimizer, visual_prompt, label_mapping, train_loader, state, args)
        masks = get_masks(mask) if mask else None
        # Acc after prune
        label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
        print('mapping_sequence: ', mapping_sequence)
        test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Accuracy after prune: {test_acc:.4f}')
        all_results['no_train_acc'] = test_acc
        # train
        if args.prune_method in ('vpns', 'bip'):
            for epoch in range(args.epochs):
                train_acc = train(train_loader, val_loader, network, epoch, label_mapping, visual_prompt, mask, args=args,
                                weight_optimizer=weight_optimizer, vp_optimizer=vp_optimizer, score_optimizer=score_optimizer, 
                                weight_scheduler=weight_scheduler, vp_scheduler=vp_scheduler, score_scheduler=score_scheduler)
                val_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
                all_results['train_acc'].append(train_acc)
                all_results['val_acc'].append(val_acc)
                # Save CKPT
                checkpoint = {
                    'state_dict': network.state_dict()
                    ,'init_weight': pre_state_init
                    ,'mask': masks
                    ,"weight_optimizer": weight_optimizer.state_dict() if weight_optimizer else None
                    ,'weight_scheduler': weight_scheduler.state_dict() if weight_scheduler else None
                    ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
                    ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
                    ,"score_optimizer": score_optimizer.state_dict() if score_optimizer else None
                    ,'score_scheduler': score_scheduler.state_dict() if score_scheduler else None
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


def init_gradients(weight_optimizer, vp_optimizer, score_optimizer):
    if weight_optimizer:
        weight_optimizer.zero_grad()
    if vp_optimizer:
        vp_optimizer.zero_grad()
    if score_optimizer:
        score_optimizer.zero_grad()


def train(train_loader, val_loader, network, epoch, label_mapping, visual_prompt, mask, args, weight_optimizer, vp_optimizer, score_optimizer, weight_scheduler, vp_scheduler, score_scheduler):
    
    def update_visual_prompt(visual_prompt, x, y):
        if args.current_steps % args.step_division==0:
            fx = label_mapping(network(visual_prompt(x)))
            loss = F.cross_entropy(fx, y, reduction='mean')
            init_gradients(weight_optimizer, vp_optimizer, score_optimizer)
            loss.backward()
            vp_optimizer.step()
        # TODO whether update label mapping

    def calculate_fx(phase, x):
        if visual_prompt and (args.global_vp_data or args.prune_mode in phase):
            fx = label_mapping(network(visual_prompt(x)))
        elif args.prune_method == 'vpns':
            fx = label_mapping(network(args.normalize(x)))
        else:
            fx = label_mapping(network(x))
        return fx
    
    # switch to train mode
    if visual_prompt:
        visual_prompt.train()
    network.train()
    start = time.time()
    total_num = 0
    true_num = 0
    loss_sum = 0

    for i, (train_batch, val_batch) in enumerate(zip(train_loader, val_loader)):
        args.current_steps+=1
        train_x, train_y = train_batch[0].cuda(), train_batch[1].cuda()
        val_x, val_y = val_batch[0].cuda(), val_batch[1].cuda()
        
        # finetune
        switch_to_finetune(network)
        phase = ['weight+vp_score', 'score+vp_weight+vp']
        fx = calculate_fx(phase, val_x)
        loss = F.cross_entropy(fx, val_y, reduction='mean')
        init_gradients(weight_optimizer, vp_optimizer, score_optimizer)
        loss.backward()
        weight_optimizer.step()
        if visual_prompt and args.prune_mode in phase:
            if args.current_steps % args.step_division==0:
                update_visual_prompt(visual_prompt, val_x, val_y)

        # bilevel
        switch_to_bilevel(network)
        phase = ['score+weight_vp']
        fx = calculate_fx(phase, train_x)
        loss = F.cross_entropy(fx, train_y, reduction='mean')
        init_gradients(weight_optimizer, vp_optimizer, score_optimizer)
        loss.backward()

        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                if param.grad is not None:  # Check if gradient exists
                    grad_vec.append(param.grad.view(-1).detach())
                else:
                    grad_vec.append(torch.zeros(param.data.numel()).to(param.device).detach())
                # grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
        
        param_grad_vec = grad2vec(network.parameters())
        
        if visual_prompt and args.prune_mode in phase:
            if args.current_steps % args.step_division==0:
                vp_optimizer.step()

        # prune
        switch_to_prune(network)
        phase = ['score+vp_weight','score+vp_weight+vp']
        fx = calculate_fx(phase, train_x)
        loss = F.cross_entropy(fx, train_y, reduction='mean')
        init_gradients(weight_optimizer, vp_optimizer, score_optimizer)
        loss.backward()

        mask_grad_vec = grad2vec(network.parameters())
        implicit_gradient = -args.bil_lr * mask_grad_vec * param_grad_vec

        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                if param.grad is not None:
                    param.grad = param.grad + vec[pointer:pointer + num_param].view_as(param).data
                else:
                    param.grad = torch.zeros_like(param.data).to(param.device)
                    param.grad = param.grad + vec[pointer:pointer + num_param].view_as(param).data
                # param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, network.parameters())
        score_optimizer.step()
        # reset threshold   
        set_prune_threshold(network, args.density)

        if visual_prompt and args.prune_mode in phase:
            if args.current_steps % args.step_division==0:
                update_visual_prompt(visual_prompt, train_x, train_y)



        total_num += train_y.size(0)
        true_num += torch.argmax(fx, 1).eq(train_y).float().sum().item()
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
    if weight_scheduler:
        print('weight_lr: ', weight_optimizer.param_groups[0]['lr'])
        weight_scheduler.step()
    if vp_scheduler:
        print('vp_lr: ', vp_optimizer.param_groups[0]['lr'])
        vp_scheduler.step()
    if score_scheduler:
        print('score_lr: ', score_optimizer.param_groups[0]['lr'])
        score_scheduler.step()

    return train_acc


def evaluate(test_loader, network, label_mapping, visual_prompt):

    def calculate_fx(phase, x):
        if visual_prompt and (args.global_vp_data or args.prune_mode in phase):
            fx = label_mapping(network(visual_prompt(x)))
        elif args.prune_method == 'vpns':
            fx = label_mapping(network(args.normalize(x)))
        else:
            fx = label_mapping(network(x))
        return fx
    
    # switch to evaluate mode
    if visual_prompt:
        visual_prompt.eval()
    network.eval()
    total_num = 0
    true_num = 0
    loss_sum = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        # compute output

        with torch.no_grad():
            phase = ['weight+vp_score', 'score+vp_weight+vp']
            fx = calculate_fx(phase, x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        # measure accuracy and record loss
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        loss_sum += loss.item() * fx.size(0)
        if (i+1) % args.print_freq == 0:
            print(f'evaluate: [{i}/{len(test_loader)}]\t'
                f'Loss_sum {loss_sum:.4f}\t'
                f'Accuracy {test_acc:.4f}\t'
            )
    print(f'evaluate: [{i}/{len(test_loader)}]\t'
        f'Loss_sum {loss_sum:.4f}\t'
        f'Accuracy {test_acc:.4f}\t'
    )
    print(f'evaluate_accuracy {test_acc:.3f}')

    return test_acc


def prune_network(network, weight_optimizer, visual_prompt, label_mapping, train_loader, state, args):
    mask = None
    args.density = args.density_list[state]
    if args.prune_method in ('random', 'omp', 'grasp','snip','synflow'):
        print(f'{(args.prune_method).upper()} pruning')
        decay = CosineDecay(args.death_rate, len(train_loader)*args.epochs)
        mask = Masking(weight_optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                    redistribution_mode=args.redistribution, args=args, train_loader=train_loader, visual_prompt=visual_prompt, label_mapping=label_mapping)
        mask.add_module(network, density=args.density_list[state], sparse_init=args.prune_method)
    elif args.prune_method in ('vpns', 'bip', 'hydra') and state > 0:
        print('Network Density Setting:', args.density)
        set_prune_threshold(network, args.density)

    return network, mask


def init_ckpt_vp_optimizer(network, visual_prompt_init, mapping_sequence, masks, args):
    best_acc = 0.
    all_results={}
    all_results['train_acc'] = []
    all_results['val_acc'] = []
    visual_prompt, score_optimizer, score_scheduler, vp_optimizer, vp_scheduler, weight_optimizer, weight_scheduler = setup_optimizer_and_prompt(network, args)
    visual_prompt.load_state_dict(visual_prompt_init) if visual_prompt_init else None
    checkpoint = {
            'state_dict': network.state_dict()
            ,'init_weight': None
            ,'mask': masks
            ,"weight_optimizer": weight_optimizer.state_dict() if weight_optimizer else None
            ,'weight_scheduler': weight_scheduler.state_dict() if weight_scheduler else None
            ,"vp_optimizer": vp_optimizer.state_dict() if vp_optimizer else None
            ,'vp_scheduler': vp_scheduler.state_dict() if vp_scheduler else None
            ,"score_optimizer": score_optimizer.state_dict() if score_optimizer else None
            ,'score_scheduler': score_scheduler.state_dict() if score_scheduler else None
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence
            ,"val_best_acc": 0
            ,'ckpt_test_acc': 0
            ,'all_results': all_results
            ,"epoch": 0
            ,'state': 0
        }

    return visual_prompt, score_optimizer, score_scheduler, vp_optimizer, vp_scheduler, weight_optimizer, weight_scheduler, checkpoint, best_acc, all_results


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
    