from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import random
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import core
from core import Masking, CosineDecay
from utils import *
from algorithms import generate_label_mapping_by_frequency, label_mapping_base, generate_label_mapping_by_frequency_ordinary

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

def main():
    parser = argparse.ArgumentParser(description='PyTorch OMP/Grasp Experiments')
    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='dataset/cifar10', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--input_size', type=int, default=32, help='size of input images')

    ##################################### Architecture ############################################
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")
    ##
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--label_mapping_mode', type=str, default='flm')

    ##################################### General setting ############################################
    parser.add_argument('--seed', default=17, type=int, help='random seed')
    parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='results/sparsity/omp_grasp', type=str)
    ##
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt', help='path to save the final model')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--max-threads', type=int, default=4, help='How many threads to use for data loading.')

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='60,90', help='decreasing strategy')
    ##
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 128)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')

    ##################################### Pruning setting #################################################
    parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
    parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
    parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
    parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')
    ##
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    # TODO what's the meaning of these parameters
    parser.add_argument('--sparse', default='true', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', default='true', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='GraSP', help='sparse initialization')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate used for dynamic sparse training (not used in this paper).')
    parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')

    # ITOP settings
    # core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print_and_log('\n\n')
    print_and_log('='*80)

    # fix random seed for Reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, train_loader, valid_loader, test_loader = setup_model_dataset(args)
    model.cuda(device)
    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    mapping_sequence = generate_label_mapping_by_frequency_ordinary(model, train_loader, device=args.gpu)
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    # if args.dataset == 'mnist':
    #     train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    # elif args.dataset == 'cifar10':
    #     # train_loader, valid_loader, test_loader = cifar10_dataloaders(args.batch_size, num_workers=args.max_threads)
    #     model, train_loader, valid_loader, test_loader = setup_model_dataset(args)
    #     model.cuda(device)
    #     output = 10
    # elif args.dataset == 'cifar100':
    #     train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
    #     output = 100

    # if args.scaled:
    #     init_type = 'scaled_kaiming_normal'
    # else:
    #     init_type = 'kaiming_normal'
    # if 'vgg' in args.model:
    #     model = vgg.VGG(depth=int(args.model[-2:]), dataset=args.data, batchnorm=True).to(device)
    # else:
    #     model = cifar_resnet.Model.get_model_from_name(args.model, initializers.initializations(init_type, args.density), outputs=output).to(device)

    print_and_log(model)
    print_and_log('=' * 60)
    print_and_log(args.model)
    print_and_log('=' * 60)
    print_and_log('=' * 60)
    # TODO what's the meaning of these parameters
    print_and_log('Prune mode: {0}'.format(args.death))
    print_and_log('Growth mode: {0}'.format(args.growth))
    print_and_log('Redistribution mode: {0}'.format(args.redistribution))
    print_and_log('=' * 60)

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer.')
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            print_and_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_and_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            print_and_log('Testing...')
            evaluate(args, model, device, test_loader)
            model.feats = []
            model.densities = []
            # plot_class_feature_histograms(args, model, device, train_loader, optimizer)
        else:
            print_and_log("=> no checkpoint found at '{}'".format(args.resume))
    if args.fp16:
        print('FP16')
        optimizer = FP16_Optimizer(optimizer, static_loss_scale = None, dynamic_loss_scale = True, dynamic_loss_args = {'init_scale': 2 ** 16})
        model = model.half()

    test_tacc = evaluate(args, model, device, valid_loader, label_mapping=label_mapping)
    print('accuracy without finetuning: ', test_tacc)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args,train_loader=train_loader)
            mask.add_module(model, label_mapping=label_mapping, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0
        # create output file
        save_path = './save/' + str(args.model) + '/' + str(args.data) + '/' + str(args.sparse_init) + '/' + str(args.seed)
        if args.sparse: save_subfolder = os.path.join(save_path, 'sparsity' + str(1 - args.density))
        else: save_subfolder = os.path.join(save_path, 'dense')
        if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)


        for epoch in range(1, args.epochs*args.multiplier + 1):

            t0 = time.time()
            if args.label_mapping_mode == 'ilm':
                mapping_sequence = generate_label_mapping_by_frequency_ordinary(model, train_loader, device=args.gpu)
                label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
            train(args, model, device, train_loader, optimizer, epoch, label_mapping=label_mapping, mask=mask)
            lr_scheduler.step()
            val_acc = evaluate(args, model, device, valid_loader, label_mapping=label_mapping)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_subfolder, 'model_final.pth'))

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        print('Testing model')
        model.load_state_dict(torch.load(os.path.join(save_subfolder, 'model_final.pth'))['state_dict'])
        evaluate(args, model, device, test_loader, label_mapping=label_mapping, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

# gradient_norm = []
def train(args, model, device, train_loader, optimizer, epoch, label_mapping, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    # global gradient_norm
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = F.log_softmax(label_mapping(model(data)), dim=1)
        loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))
    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))


def evaluate(args, model, device, test_loader, label_mapping, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = F.log_softmax(label_mapping(model(data)), dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


if __name__ == '__main__':
   main()
