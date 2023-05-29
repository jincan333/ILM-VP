import torch
from torch import nn
from torch.nn import functional as F
import argparse
from torch.nn.utils import prune
from functools import partial
import os

# from algorithms import label_mapping_base, generate_label_mapping_by_frequency_ordinary
# from parser import add_args, load_args
# from utils import set_seed, setup_model_dataset
# from pruner import 

# def main(ckpt_path):
#     # Parser
#     parser = argparse.ArgumentParser(description='Evaluate Model Effect')
#     global args
#     args = parser.parse_args()
#     # Device
#     device = torch.device(f"cuda:{args.gpu}")
#     torch.cuda.set_device(int(args.gpu))
#     set_seed(args.seed)
#     network, train_loader, val_loader, test_loader = setup_model_dataset(args)
#     network.cuda()

#     ckpt = torch.load(ckpt_path, map_location = torch.device('cuda:'+str(args.gpu)))
#     visual_prompt = ckpt['visual_prompt']
#     mapping_sequence = ckpt['mapping_sequence']

#     ,"optimizer_dict": optimizer.state_dict()
#     ,'scheduler': scheduler.state_dict()
#     ,'visual_prompt': visual_prompt.state_dict() if visual_prompt else None
#     ,'mapping_sequence': mapping_sequence if args.label_mapping_mode else None
#     ,"epoch": epoch
#     ,"val_best_acc": best_acc
#     ,'all_results': all_results
#     ,'init_weight': initalization
#     ,'state': state

#         test_acc, test_loss = evaluate(test_loader, network, label_mapping)
#         acc_list.append(test_acc)
#         print(f'[{i}] Checkpoint path: {ckpt_path}\t'
#             ,f'Accuracy:{test_acc:.4f}'
#         )
#     print('Accuracy list: ', acc_list)


# def load_ckpt(network, ckpt_path):
#     checkpoint = torch.load(ckpt_path, map_location = torch.device('cuda:'+str(args.gpu)))

#     if args.is_prune:
#         current_mask = extract_mask(checkpoint['state_dict'])
#         prune_model_custom(network, current_mask)
#         check_sparsity(network)
#         network.load_state_dict(checkpoint['state_dict'])
#         # adding an extra forward process to enable the masks
#         x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
#         network.eval()
#         with torch.no_grad():
#             network(x_rand)
#     else:
#         network.load_state_dict(checkpoint['state_dict'])

#     return network



# def evaluate(val_loader, model, label_mapping):
#     # switch to evaluate mode
#     model.eval()
#     total_num = 0
#     true_num = 0
#     loss_sum = 0
#     for i, (x, y) in enumerate(val_loader):
#         x = x.cuda()
#         y = y.cuda()
#         # compute output
#         with torch.no_grad():
#             if label_mapping:
#                 fx = label_mapping(model(x))
#             else:
#                 fx = model(x)
#             loss = F.cross_entropy(fx, y, reduction='mean')
#         # measure accuracy and record loss
#         total_num += y.size(0)
#         true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
#         test_acc = true_num / total_num
#         loss_sum += loss.item() * fx.size(0)
#         if i % args.print_freq == 0:
#             print(f'evaluate: [{i}/{len(val_loader)}]\t'
#                 f'Loss {loss_sum:.4f}\t'
#                 f'Accuracy {test_acc:.4f}\t'
#             )
#     print(f'evaluate_accuracy {test_acc:.3f}')

#     return test_acc, loss_sum



if __name__ == '__main__':
    dir_path = '/data4/hop20001/can/ILM-VP/results/resnet18/cifar10/VPFalse/PRUNEimp/LMflm/LPFalse/sgd/LR0.01/cosine/EPOCHS200/IMAGESIZE32/GPU5/ff_prune'
    ckpt_path = []
    acc = []
    for i in range(1):
        ckpt_path.append(os.path.join(dir_path, str(i) + 'best.pth'))
    for path in ckpt_path:
        acc.append(torch.load(path)['ckpt_test_acc'])
    print(acc)
