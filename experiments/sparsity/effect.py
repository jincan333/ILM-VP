import torch
from torch import nn
from utils import *
from pruner import *
import argparse
from torch.nn.utils import prune

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='data/cifar10', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=17, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='results/imp/sparsity', type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
# parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=9, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

global args
args = parser.parse_args()

model, train_loader, val_loader, test_loader = setup_model_dataset(args)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
acc_list = []
for i in range(1,9):
    checkpoint_path = f'results/imp/sparsity/{i}checkpoint.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location = torch.device('cuda:'+str(args.gpu)))
    best_sa = checkpoint['best_sa']
    start_epoch = checkpoint['epoch']
    all_result = checkpoint['result']
    start_state = checkpoint['state']

    if start_state>0:
        current_mask = extract_mask(checkpoint['state_dict'])
        prune_model_custom(model, current_mask)
        check_sparsity(model)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    model.load_state_dict(checkpoint['state_dict'])
    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    initalization = checkpoint['init_weight']
    print('loading state:', start_state)
    print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)
    test_tacc = validate(test_loader, model, criterion)
    acc_list.append(test_tacc)
print('test_accuracy list: ', acc_list)




