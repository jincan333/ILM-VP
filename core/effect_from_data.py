import torch
import os
import sys
sys.path.append('./core')
import argparse
import torch
from torch.nn import functional as F
from utils import set_seed, setup_optimizer_and_prompt, calculate_label_mapping, obtain_label_mapping, save_args, get_masks
from get_model_dataset import choose_dataloader, get_model
from core import Masking, CosineDecay
from pruner import extract_mask, prune_model_custom, check_sparsity, remove_prune, pruning_model
from hydra import set_hydra_prune_rate, set_hydra_network


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Visual Prompt + Prune Experiments')
    global args
    parser.add_argument('--prune_mode', type=str, default='normal', choices=['normal', 'vp_ff', 'no_tune', 'vp'], help='prune method implement ways')
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
    parser.add_argument('--dataset', default="tiny_imagenet", choices=['cifar10', 'cifar100', 'flowers102', 'dtd', 'food101', 'oxfordpets', 'stanfordcars', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--experiment_name', default='exp', type=str, help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', default=120, type=int, help='number of total eopchs to run')
    parser.add_argument('--seed', default=17, type=int, help='random seed')
    parser.add_argument('--density_list', default='1,0.10,0.01,0.001', type=str, help='density list(1-sparsity), choose from 1,0.50,0.40,0.30,0.20,0.10,0.05')
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
    device = torch.device(f"cuda:{args.gpu}")
    args.device=device
    torch.cuda.set_device(int(args.gpu))
    args.prompt_method=None if args.prompt_method=='None' else args.prompt_method
    args.density_list=[float(i) for i in args.density_list.split(',')]
    set_seed(args.seed)
    dir_path = '/data4/hop20001/can/ILM-VP/result/main_normal/resnet18/tiny_imagenet/PRUNE_MODEnormal/PRUNEhydra/VPpad/LMflm/SIZE224_224_16_183/adam_adam_adam/cosine_multistep_cosine/LR0.001_0.001_0.0001/DENSITY[1.0, 0.1, 0.01, 0.001]/EPOCHS120/SEED17/GPU7'
    # with open(dir_path, 'rb') as file:
    #     data = pickle.load(file)
    # print(file)
    ckpt_path = []
    acc = []
    for i in range(1,4):
        ckpt_path.append(os.path.join(dir_path, str(i) + 'best.pth'))
    for i,path in enumerate(ckpt_path):
        network = get_model(args)
        network = set_hydra_network(network, args)
        phase = 'subnetwork'
        train_loader, val_loader, test_loader = choose_dataloader(args, phase)
        visual_prompt, hydra_optimizer, hydra_scheduler, vp_optimizer, vp_scheduler, ff_optimizer, ff_scheduler = setup_optimizer_and_prompt(network, args)
        best_ckpt = torch.load(path)
        network.load_state_dict(best_ckpt['state_dict'])
        visual_prompt.load_state_dict(best_ckpt['visual_prompt']) if visual_prompt else None
        mapping_sequence = best_ckpt['mapping_sequence']
        label_mapping = obtain_label_mapping(mapping_sequence)
        network, mask = prune_network(network, ff_optimizer, visual_prompt, label_mapping, train_loader, i+1, args)
        test_acc = evaluate(test_loader, network, label_mapping, visual_prompt)
        print(f'Best CKPT Accuracy: {test_acc:.4f}')
        acc.append(test_acc)
    print(acc)




