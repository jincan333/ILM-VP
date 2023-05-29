import math
import warnings

from torch import Tensor
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.nn.functional as F



def replace_layers(model, old_layer, new_layer,channelPrune):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_layers(module, old_layer, new_layer,channelPrune)

        if type(module) == old_layer:
            if isinstance(module, nn.Conv2d):
                layer_new = new_layer(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias,channelPrune)
            elif isinstance(module, nn.Linear):
                layer_new = new_layer(module.in_features, module.out_features, module.bias,channelPrune)
            layer_new.weight.data=module.weight.data
            if module.bias is not None:
                layer_new.bias.data=module.bias.data
            model._modules[name] = layer_new

    return model


def setPruneRate(network,cl,ll,r):
    for name, module in network.named_modules():
        if isinstance(module,cl):
            module.set_prune_rate(r)
        if  isinstance(module,ll):
            module.set_prune_rate(r)



def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear
    else:
        raise ValueError("Incorrect layer type")


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        channel_prune='kernel'
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.channel_prune=channel_prune
        if channel_prune=='kernel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],self.weight.shape[1],1,1)))
        elif channel_prune=='channel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],1,1,1)))
        elif channel_prune=='inputchannel':
            self.popup_scores = Parameter(torch.randn((1,self.weight.shape[1],self.weight.shape[2],self.weight.shape[3])))
        else:
            self.popup_scores = Parameter(torch.randn(self.weight.shape))
        self.popup_scores.is_score=True
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.k=False
    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        if self.k is False:
            adj=1.0
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def __init__(self, in_features, out_features, bias=True,channel_prune='kernel'):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.channel_prune=channel_prune
        if channel_prune=='channel':
            self.popup_scores = Parameter(torch.randn((self.weight.shape[0],1)))
        elif channel_prune=='inputchannel':
            self.popup_scores = Parameter(torch.randn((1,self.weight.shape[1])))
        else:
            self.popup_scores = Parameter(torch.randn(self.weight.shape))
        self.popup_scores.is_score=True
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        self.k=False

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        if self.k is False:
            adj=1.0
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x




# def freeze_vars(model, var_name, freeze_bn=False):
#     """
#     freeze vars. If freeze_bn then only freeze batch_norm params.
#     """

#     assert var_name in ["weight", "bias", "popup_scores"]
#     for i, v in model.named_modules():
#         if hasattr(v, var_name):
#             if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
#                 if getattr(v, var_name) is not None:
#                     getattr(v, var_name).requires_grad = False


# def unfreeze_vars(model, var_name):
#     assert var_name in ["weight", "bias", "popup_scores"]
#     for i, v in model.named_modules():
#         if hasattr(v, var_name):
#             if getattr(v, var_name) is not None:
#                 getattr(v, var_name).requires_grad = True


# def set_prune_rate_model(model, prune_rate):
#     for _, v in model.named_modules():
#         if hasattr(v, "set_prune_rate"):
#             v.set_prune_rate(prune_rate)


# def get_layers(layer_type):
#     """
#         Returns: (conv_layer, linear_layer)
#     """
#     if layer_type == "dense":
#         return nn.Conv2d, nn.Linear
#     elif layer_type == "subnet":
#         return SubnetConv, SubnetLinear
#     else:
#         raise ValueError("Incorrect layer type")


# def show_gradients(model):
#     for i, v in model.named_parameters():
#         print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


# def snip_init(model, criterion, optimizer, train_loader, device, args):
#     print("Using SNIP initialization")
#     assert args.exp_mode == "pretrain"
#     optimizer.zero_grad()
#     # init the score with kaiming normal init
#     for m in model.modules():
#         if hasattr(m, "popup_scores"):
#             nn.init.kaiming_normal_(m.popup_scores, mode="fan_in")

#     set_prune_rate_model(model, 1.0)
#     unfreeze_vars(model, "popup_scores")

#     # take a forward pass and get gradients
#     for _, data in enumerate(train_loader):
#         images, target = data[0].to(device), data[1].to(device)

#         output = model(images)
#         loss = criterion(output, target)

#         loss.backward()
#         break

#     # update scores with their respective connection sensitivty
#     for m in model.modules():
#         if hasattr(m, "popup_scores"):
#             print(m.popup_scores.data)
#             m.popup_scores.data = m.popup_scores.grad.data.abs()
#             print(m.popup_scores.data)

#     # update k back to args.k.
#     set_prune_rate_model(model, args.k)
#     freeze_vars(model, "popup_scores")


# def initialize_scores(model, init_type):
#     print(f"Initialization relevance score with {init_type} initialization")
#     for m in model.modules():
#         if hasattr(m, "popup_scores"):
#             if init_type == "kaiming_uniform":
#                 nn.init.kaiming_uniform_(m.popup_scores)
#             elif init_type == "kaiming_normal":
#                 nn.init.kaiming_normal_(m.popup_scores)
#             elif init_type == "xavier_uniform":
#                 nn.init.xavier_uniform_(
#                     m.popup_scores, gain=nn.init.calculate_gain("relu")
#                 )
#             elif init_type == "xavier_normal":
#                 nn.init.xavier_normal_(
#                     m.popup_scores, gain=nn.init.calculate_gain("relu")
#                 )




# def scale_rand_init(model, k):
#     print(
#         f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
#     )
#     for m in model.modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             # print(f"previous std = {torch.std(m.weight.data)}")
#             m.weight.data = 1 / math.sqrt(k) * m.weight.data
#             # print(f"new std = {torch.std(m.weight.data)}")


# def prepare_model(model, args):
#     """
#         1. Set model pruning rate
#         2. Set gradients base of training mode.
#     """

#     set_prune_rate_model(model, args.k)

#     if args.exp_mode == "pretrain":
#         print(f"#################### Pre-training network ####################")
#         print(f"===>>  gradient for importance_scores: None  | training weights only")
#         freeze_vars(model, "popup_scores", args.freeze_bn)
#         unfreeze_vars(model, "weight")
#         unfreeze_vars(model, "bias")

#     elif args.exp_mode == "prune":
#         print(f"#################### Pruning network ####################")
#         print(f"===>>  gradient for weights: None  | training importance scores only")

#         unfreeze_vars(model, "popup_scores")
#         freeze_vars(model, "weight", args.freeze_bn)
#         freeze_vars(model, "bias", args.freeze_bn)

#     elif args.exp_mode == "finetune":
#         print(f"#################### Fine-tuning network ####################")
#         print(
#             f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
#         )
#         freeze_vars(model, "popup_scores", args.freeze_bn)
#         unfreeze_vars(model, "weight")
#         unfreeze_vars(model, "bias")

#     else:
#         assert False, f"{args.exp_mode} mode is not supported"

#     initialize_scores(model, args.scores_init_type)


# def subnet_to_dense(subnet_dict, p):
#     """
#         Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
#         loaded in network with dense layers.
#     """
#     dense = {}

#     # load dense variables
#     for (k, v) in subnet_dict.items():
#         if "popup_scores" not in k:
#             dense[k] = v

#     # update dense variables
#     for (k, v) in subnet_dict.items():
#         if "popup_scores" in k:
#             s = torch.abs(subnet_dict[k])

#             out = s.clone()
#             _, idx = s.flatten().sort()
#             j = int((1 - p) * s.numel())

#             flat_out = out.flatten()
#             flat_out[idx[:j]] = 0
#             flat_out[idx[j:]] = 1
#             dense[k.replace("popup_scores", "weight")] = (
#                 subnet_dict[k.replace("popup_scores", "weight")] * out
#             )
#     return dense


# def dense_to_subnet(model, state_dict):
#     """
#         Load a dict with dense-layer in a model trained with subnet layers. 
#     """
#     model.load_state_dict(state_dict, strict=False)


# def current_model_pruned_fraction(model, result_dir, verbose=True):
#     """
#         Find pruning raio per layer. Return average of them.
#         Result_dict should correspond to the checkpoint of model.
#     """

#     # load the dense models
#     path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

#     pl = []

#     if os.path.exists(path):
#         state_dict = torch.load(path, map_location="cpu")["state_dict"]
#         for i, v in model.named_modules():
#             if isinstance(v, (nn.Conv2d, nn.Linear)):
#                 if i + ".weight" in state_dict.keys():
#                     d = state_dict[i + ".weight"].data.cpu().numpy()
#                     p = 100 * np.sum(d == 0) / np.size(d)
#                     pl.append(p)
#                     if verbose:
#                         print(i, v, p)
#         return np.mean(pl)


# def sanity_check_paramter_updates(model, last_ckpt):
#     """
#         Check whether weigths/popup_scores gets updated or not compared to last ckpt.
#         ONLY does it for 1 layer (to avoid computational overhead)
#     """
#     for i, v in model.named_modules():
#         if hasattr(v, "weight") and hasattr(v, "popup_scores"):
#             if getattr(v, "weight") is not None:
#                 w1 = getattr(v, "weight").data.cpu()
#                 w2 = last_ckpt[i + ".weight"].data.cpu()
#             if getattr(v, "popup_scores") is not None:
#                 s1 = getattr(v, "popup_scores").data.cpu()
#                 s2 = last_ckpt[i + ".popup_scores"].data.cpu()
#             return not torch.allclose(w1, w2), not torch.allclose(s1, s2)




