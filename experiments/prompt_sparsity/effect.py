import torch
from torch import nn
from torch.nn import functional as F
import argparse
from torch.nn.utils import prune
from functools import partial
import os


if __name__ == '__main__':
    dir_path = '/data4/hop20001/can/ILM-VP/results/resnet18/cifar10/ablation_no_sigmoid_exp/VPpad/PRUNEimp/LMilm/LPFalse/adam/LR0.01/multistep/EPOCHS200/IMAGESIZE128_48_156/SEED7/GPU5/DENSITY0.8'
    ckpt_path = []
    acc = []
    for i in range(0,10):
        ckpt_path.append(os.path.join(dir_path, str(i) + 'best.pth'))
    for path in ckpt_path:
        acc.append(torch.load(path)['ckpt_test_acc'])
    print(acc)

# 0609
# ablation_lm_methods_exp
flm = [0.7425, 0.7348, 0.7421, 0.7325, 0.728, 0.7253, 0.718, 0.6968, 0.694, 0.666]

# ablation_hydra_init_exp
init_1 = [0.7948, 0.9407, 0.9319, 0.9148, 0.8877, 0.8754, 0.8754, 0.8641, 0.8549, 0.8427]
init_0 = [0.7948, 0.9398, 0.9238, 0.9092, 0.8918, 0.8681, 0.8674, 0.8614, 0.8569, 0.8484]

# ablation_randomcrop_exp
crop_0=[0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
crop_1=[0.8358, 0.8212, 0.8183, 0.7219, 0.7391, 0.6887, 0.6877, 0.6386, 0.6599, 0.6235]

# 0608
# ablation_hydra_init_exp
init_1 = [0.9497, 0.9395, 0.9263, 0.9059, 0.8928]
init_0 = [0.9489, 0.9364, 0.9222, 0.9002, 0.8836]

# ablation_randomcrop_exp
crop_0=[0.8152, 0.8186, 0.8265, 0.801, 0.7658]
crop_1=[0.8271, 0.8174, 0.818, 0.7829, 0.7645]

# 0605
# prompt_prune_exp
imp_flm = [0.7401, 0.744, 0.738, 0.7435, 0.7239, 0.7185, 0.7072, 0.6968, 0.682, 0.6548]
imp_ilm = [0.8051, 0.8267, 0.813, 0.8159, 0.7851, 0.6392, 0.6771, 0.6365, 0.6127, 0.6011]
omp_flm = [0.7401, 0.7415, 0.7391, 0.734, 0.7254, 0.71, 0.7018, 0.6998, 0.662, 0.6314]
omp_ilm = [0.8051, 0.8308, 0.7939, 0.7929, 0.7799, 0.6419, 0.6186, 0.6202, 0.6103, 0.5916]
grasp_flm = [0.7401, 0.6009, 0.5669, 0.5299, 0.2504, 0.1938, 0.2389, 0.2993, 0.1836, 0.2076]
grasp_ilm = [0.8051, 0.5622, 0.5446, 0.5231, 0.4542, 0.4376, 0.3994, 0.3692, 0.3823, 0.3865]
hydra_flm = [0.7401, 0.9491, 0.9395, 0.9157, 0.905, 0.884, 0.8761, 0.8713, 0.866, 0.864]
hydra_ilm = [0.8051, 0.9489, 0.9364, 0.9222, 0.9002, 0.8836, 0.8768, 0.8759, 0.8732, 0.8649]

# tricks_exp_flm_after_prune_notune
imp = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
omp = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
grasp = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]


# tricks_exp_flm_after_prune
omp_flm_pre = [0.4614, 0.7415, 0.7391]
omp_flm_after = [0.4616, 0.7378, 0.75]
omp_ilm = [0.4614, 0.8308, 0.7939]
grasp_flm_pre = [0.4616, 0.6009, 0.5669]
grasp_flm_after = [0.4616, 0.5506, 0.522]
grasp_ilm = [0.4616, 0.5622, 0.5446]

# prompt_loc_exp
pad = [0.6286, 0.6224]
fix = [0.5754, 0.5814]
random = [0.4755, 0.4678]






# 0604
input_32 = [0.5522, 0.5565, 0.5599, 0.5636, 0.5581]
input_64 = [0.6043, 0.6007, 0.6074, 0.6046, 0.5961]
input_96 = [0.6374, 0.6385, 0.6403, 0.6382, 0.635]
input_128 = [0.6492, 0.6542, 0.661, 0.6453, 0.6454]
input_160 = [0.7401, 0.744, 0.738, 0.7435, 0.7239]
input_192 = [0.6773, 0.6789, 0.68, 0.6758, 0.6689]
input_224 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]



pad_112 = [0.6179, 0.6147, 0.6077, 0.6173, 0.6145]
pad_96 = [0.6118, 0.6199, 0.6159, 0.6226, 0.6082]
pad_80 = [0.6166, 0.6202, 0.6142, 0.6115, 0.6159]
pad_64 = [0.6135, 0.6112, 0.6037, 0.6072, 0.61]
pad_48 = [0.614, 0.6055, 0.6066, 0.6072, 0.5923]
pad_32 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]
pad_16 = [0.5896, 0.5918, 0.5919, 0.5892, 0.5845]
# pad_16 0.5894
# pad_32 0.6157600000000001
# pad_64 0.60912
# pad_80 0.61568
# pad_96 0.61568
# pad_112 0.6144200000000001

# 0603
# prune without tune
imp_flm = [0.5042, 0.4983, 0.4723, 0.4754, 0.4616, 0.455, 0.3472, 0.1568, 0.1268, 0.0971]
imp_ilm = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
omp_flm = [0.5042, 0.499, 0.475, 0.4825, 0.4797, 0.4214, 0.2975, 0.1296, 0.1028, 0.1003]
omp_ilm = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
grasp_flm = [0.5042, 0.0848, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
grasp_ilm = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]
hydra_flm = [0.5042, 0.1104, 0.1149, 0.0976, 0.1053, 0.1, 0.1, 0.1, 0.1, 0.1]
hydra_ilm = [0.5042, 0.1724, 0.1217, 0.1355, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]




# 0602
# TODO initialization may be different
imp_ff_flm = [0.9625, 0.9668, 0.9652, 0.9656, 0.9676, 0.9668, 0.9651, 0.9672, 0.9649, 0.9648]
imp_ff_ilm = [0.9625, 0.9666, 0.9682, 0.9655, 0.966, 0.9661, 0.9645, 0.9638, 0.9652, 0.962]
omp_ff_flm = [0.9625, 0.9633, 0.9627, 0.9615, 0.9604, 0.9586, 0.9581, 0.9582, 0.9563, 0.9552]
omp_ff_ilm = [0.9625, 0.9632, 0.9635, 0.9621, 0.9631, 0.9586, 0.9592, 0.9582, 0.9541, 0.9529]
grasp_ff_flm = [0.9625, 0.9602, 0.9585, 0.9532, 0.9526, 0.9451, 0.9406, 0.9383, 0.9386, 0.9322]
grasp_ff_ilm = [0.9625, 0.9599, 0.9544, 0.9511, 0.9483, 0.9453, 0.944, 0.9399, 0.9375, 0.9321]
hydra_ff_flm = [0.9625, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]
hydra_ff_ilm = [0.9625, 0.9636, 0.9579, 0.9564, 0.9449, 0.9363, 0.9288, 0.9258, 0.923, 0.9193]

imp_vp_flm = [0.6528, 0.6181, 0.617, 0.6218, 0.6146, 0.6123, 0.6003, 0.5842, 0.5638, 0.5361]
imp_vp_ilm = [0.66, 0.6164, 0.6144, 0.6151, 0.6131, 0.6008, 0.5878, 0.5705, 0.5475, 0.5183]
omp_vp_flm = [0.6528, 0.616, 0.621, 0.6132, 0.6179, 0.6121, 0.6012, 0.571, 0.5351, 0.4976]
omp_vp_ilm = [0.66, 0.6265, 0.6261, 0.6157, 0.6067, 0.5925, 0.5672, 0.549, 0.5024, 0.5011]
grasp_vp_flm = [0.6325, 0.5462, 0.5359, 0.5493, 0.5429, 0.5565, 0.5466, 0.2311, 0.2327]
grasp_vp_ilm = [0.5684, 0.5709, 0.5428, 0.5398, 0.5448, 0.5711, 0.5737, 0.4063, 0.4035]
hydra_vp_flm = [0.8399, 0.8215, 0.8102, 0.8056, 0.8002, 0.7837, 0.7844, 0.7703, 0.768]
hydra_vp_ilm = [0.8314, 0.8239, 0.809, 0.798, 0.7884, 0.7923, 0.786, 0.7725, 0.752]


pad_16 = [0.551, 0.5445, 0.5473, 0.5505, 0.5392, 0.5379, 0.521, 0.5333, 0.5246, 0.5084]
pad_32 = [0.6159, 0.6133, 0.6148, 0.613, 0.6143, 0.6064, 0.5956, 0.5845, 0.5803, 0.5809]
pad_48 = [0.6256, 0.6252, 0.6327, 0.6248, 0.6276, 0.6132, 0.6179, 0.6032, 0.6079, 0.601]
pad_64 = [0.6273, 0.6332, 0.6246, 0.6243, 0.6259, 0.6268, 0.6215, 0.6224, 0.6175, 0.6117]
pad_80 = [0.6356, 0.6414, 0.6382, 0.6296, 0.6417, 0.628, 0.6266, 0.6269, 0.6227, 0.6298]
pad_96 = [0.6442, 0.6357, 0.6494, 0.6326, 0.6388, 0.636, 0.6266, 0.6288, 0.6333, 0.624]






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


