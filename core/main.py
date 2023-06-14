import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from matplotlib import pyplot as plt
import copy
import warnings

from utils import set_seed, setup_optimizer_and_prompt, calculate_label_mapping, obtain_label_mapping, save_args
from get_model_dataset import get_torch_dataset, get_model
from pruner import extract_mask, prune_model_custom, check_sparsity, remove_prune, pruning_model
from core import Masking, CosineDecay
from hydra import set_hydra_prune_rate, set_hydra_network


def main():    
    parser = argparse.ArgumentParser(description='PyTorch Visual Prompt + Prune + Label Mapping Experiments')
    global args

    # Frequently change
    # parser.add_argument('--label_mapping_mode', type=str, default='ilm', choices=['flm', 'ilm'], required=True)
    # parser.add_argument('--prompt_method', type=str, default='pad', choices=['pad', 'fix', 'random', 'None'], required=True)
    # parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'], required=True)
    # parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy.', choices=['cosine', 'multistep'], required=True)
    # parser.add_argument('--prune_method', type=str, default='hydra', choices=['imp', 'omp', 'grasp', 'hydra'], required=True)
    # parser.add_argument('--is_finetune', default=0, type=int, choices=[0, 1], required=True)
    # parser.add_argument('--network', default='resnet18', choices=["resnet18", "resnet50", "instagram"], required=True)
    # parser.add_argument('--dataset', default="cifar10", choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], required=True)

    parser.add_argument('--label_mapping_mode', type=str, default='ilm', choices=['flm', 'ilm'])
    parser.add_argument('--prompt_method', type=str, default='pad', choices=['pad', 'fix', 'random', 'None'])
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use.', choices=['sgd', 'adam'])
    parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy.', choices=['cosine', 'multistep'])
    parser.add_argument('--prune_method', type=str, default='hydra', choices=['imp', 'omp', 'grasp', 'hydra'])
    parser.add_argument('--is_finetune', default=0, type=int, choices=[0, 1])
    parser.add_argument('--network', default='resnet18', choices=["resnet18", "resnet50", "instagram"])
    parser.add_argument('--dataset', default="cifar10", choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"])


    parser.add_argument('--experiment_name', default='exp', type=str, help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--input_size', type=int, default=128, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
    parser.add_argument('--pad_size', type=int, default=48, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[0, 16, 32, 48, 64, 80, 96, 112])
    parser.add_argument('--mask_size', type=int, default=183, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
    parser.add_argument('--epochs', default=200, type=int, help='number of total eopchs to run')
    parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
    parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--hydra_scaled_init', type=int, default=1, help='whether use scaled initialization for hydra or not.', choices=[0, 1])
    parser.add_argument('--flm_loc', type=str, default='pre', help='pre-train flm or after-prune flm.', choices=['pre', 'after'])
    parser.add_argument('--randomcrop', type=int, default=0, help='dataset randomcrop.', choices=[0, 1])
    parser.add_argument('--is_adjust_linear_head', type=int, default=0, choices=[0, 1])
    parser.add_argument('--start_state', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    ##################################### General setting ############################################
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='results', type=str)
    # parser.add_argument('--experiment_name', default='test', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
    # parser.add_argument('--gpu', type=int, default=6, help='gpu device id')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
    parser.add_argument('--resume_checkpoint', default='', help="resume checkpoint path")
    parser.add_argument('--print_freq', default=200, type=int, help='print frequency')

    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')

    ##################################### Architecture ############################################
    # parser.add_argument('--label_mapping_mode', type=str, default='flm', help='label mapping methods: rlm, flm, ilm, None')
    parser.add_argument('--label_mapping_interval', type=int, default=1, help='in ilm, the interval of epoch to implement label mapping')

    ##################################### Visual Prompt ############################################
    # parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random')
    # parser.add_argument('--input_size', type=int, default=32, help='image size before prompt, only work for exapnd')
    parser.add_argument('--output_size', type=int, default=224, help='image size after prompt, fix to 224')
    # parser.add_argument('--pad_size', type=int, default=96, help='pad size of padprompt, no more than 112, parameters cnt 4*pad**2+896pad')
    # parser.add_argument('--mask_size', type=int, default=96, help='mask size of fixadd and randomadd, no more than 224, parameters cnt mask**2')
    
    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    # parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    # parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
    parser.add_argument('--decreasing_step', default=[0.5,0.72], type = list, help='decreasing strategy')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')

    ##################################### Pruning setting #################################################
    # parser.add_argument('--prune_method', type=str, default='hydra', help='prune methods: imp, omp, grasp, hydra')
    # parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
    # parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
    parser.add_argument('--fix', default='true', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate used for dynamic sparse training (not used in this paper).')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    # IMP
    parser.add_argument('--imp_prune_type', default='pt', type=str, help='IMP type (lt, pt or rewind_lt)')
    parser.add_argument('--imp_random_prune', action='store_true', help='whether using random prune')
    parser.add_argument('--imp_rewind_epoch', default=3, type=int, help='rewind checkpoint')
    # Hydra
    parser.add_argument('--layer_type',choices=['subnet','dense'],default='subnet')
    parser.add_argument('--ChannelPrune', choices=['kernel','channel','weight','inputchannel'],default='weight')
    parser.add_argument('--init_type',default='kaiming_normal')
    parser.add_argument('--scaled_score_init',default=True)
    parser.add_argument('--k',default=0.80)
    parser.add_argument('--exp_mode',default='prune')
    parser.add_argument('--freeze_bn',default=False)
    parser.add_argument('--scores_init_type',default=None, choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"))

    args = parser.parse_args()
    if args.prompt_method=='None':
        args.prompt_method=None
    print(args)
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    args.device=device
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    # Save Path
    save_path = os.path.join(args.save_dir, args.experiment_name, args.network, args.dataset, 'VP'+str(args.prompt_method), 'FF'+str(args.is_finetune), 'LP'+str(args.is_adjust_linear_head), 
                'PRUNE'+str(args.prune_method), 'LM'+str(args.label_mapping_mode), args.optimizer, 'LR'+str(args.lr), args.lr_scheduler, 'EPOCHS'+str(args.epochs), 
                'IMAGESIZE'+str(args.input_size)+'_'+str(args.pad_size)+'_'+str(args.mask_size), 'SEED'+str(args.seed), 'GPU'+str(args.gpu), 'DENSITY'+str(args.k))
    os.makedirs(save_path, exist_ok=True)
    save_args(args, save_path+'args.pkl')
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    print('Save path: ',save_path)
    # Network and Dataset
    network = get_model(args)
    train_loader, val_loader, test_loader = get_torch_dataset(args)
    if args.is_adjust_linear_head:
        network.fc = torch.nn.Linear(512, args.class_cnt).to(args.device)
    if args.prune_method == 'hydra':
        network = set_hydra_network(network, args)
    print(network)
    # Optimizer and Scheduler
    optimizer, scheduler, visual_prompt = setup_optimizer_and_prompt(network, args)
    # Label Mapping
    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
    # Prune initiate type
    initalization = copy.deepcopy(network.state_dict())
    best_acc = 0. 
    start_state = 0
    all_results={}
    all_results['train_acc'] = []
    all_results['val_acc'] = []
    # Accuracy before train
    test_acc, test_loss = evaluate(test_loader, network, label_mapping, visual_prompt)
    print(f'Accuracy before train: {test_acc:.4f}, loss {test_loss:.4f}')
    all_results['no_train_acc'] = test_acc
    checkpoint = {
            'state_dict': network.state_dict()
            ,"optimizer_dict": optimizer.state_dict() if optimizer else None
            ,'scheduler': scheduler.state_dict() if scheduler else None
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence
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
    start_state = args.start_state if args.start_state > 0 else start_state
    # if optimizer=None, it means there are no need to finetune or visual prompt
    if not optimizer:
        epochs = 1
    else:
        epochs = args.epochs * args.multiplier
    print('######################################## Start Standard Training Iterative Pruning ########################################')
    for state in range(start_state, args.pruning_times):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        mask = None
        for epoch in range(epochs):
            if optimizer:
                print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch == args.warmup:
                # OMP, GraSP, Hydra Prune
                if args.prune_method in ('omp', 'grasp') and state > 0:
                    decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
                    mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                                redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
                    mask.add_module(network, label_mapping=label_mapping, sparse_init=args.prune_method, density=(args.density)**state)
                elif args.prune_method == 'hydra' and state > 0:
                    print('Hydra Density Setting:', (args.density)**state)
                    set_hydra_prune_rate(network, (args.density)**state, args)
                else:
                    pass
                if args.flm_loc == 'after':
                    print('Using after-prune label mapping\n')
                    label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
            # Train
            if args.label_mapping_mode == 'ilm' and epoch % args.label_mapping_interval == 0:
                label_mapping, mapping_sequence = calculate_label_mapping(visual_prompt, network, train_loader, args)
            if optimizer:
                train_acc, train_loss = train(train_loader, network, optimizer, epoch, label_mapping, visual_prompt, mask)
                scheduler.step()
            else:
                warnings.warn('\n*********No Need For Model to train!!!*********\n')
                train_acc, train_loss = 0, 10000
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
                ,"optimizer_dict": optimizer.state_dict() if optimizer else None
                ,'scheduler': scheduler.state_dict() if scheduler else None
                ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
                ,'mapping_sequence': mapping_sequence
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
        if (args.prune_method == 'imp' and args.imp_prune_type == 'pt') or args.prune_method in ('omp', 'grasp', 'hydra'):
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
        optimizer, scheduler, visual_prompt = setup_optimizer_and_prompt(network, args)
        # Save for the next iteration
        test_acc, test_loss = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Accuracy without train: {test_acc:.4f}, loss {test_loss:.4f}')
        all_results['no_train_acc'] = test_acc
        checkpoint = {
            'state_dict': network.state_dict()
            ,"optimizer_dict": optimizer.state_dict() if optimizer else None
            ,'scheduler': scheduler.state_dict() if scheduler else None
            ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
            ,'mapping_sequence': mapping_sequence
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
    