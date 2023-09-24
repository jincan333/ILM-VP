import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d
from torchvision import models


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        zero = torch.tensor([0.]).cuda()
        one = torch.tensor([1.]).cuda()
        out = torch.where(out <= threshold, zero, one)
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
        # self.weight.requires_grad = False
        # if self.bias is not None:
        #     self.bias.requires_grad = False
        self.w = 0
        self.k=False
        self.threshold=False
        self.adj=1.0
        self.pre_adj=1.0
        self.pre_scores=1.0
        self.training = True

    def set_threshold(self, threshold):
        self.threshold = threshold

    def calculate_mask(self,pre_adj,pre_scores=None):
        if type(pre_adj) is float:
            self.pre_adj=pre_adj            
        else:
            self.pre_adj=pre_adj.view(1,self.weight.data.shape[1],1,1)
        if type(pre_scores) is float:
            self.pre_scores=pre_scores
        else:
            self.pre_scores=pre_scores.view(1,self.weight.data.shape[1],1,1)

        if self.threshold is False:
            self.adj=1.0
        else:
            self.adj=GetSubnet.apply(self.popup_scores.abs(), self.threshold)
        return self.adj

    def forward(self, x):    
        if self.channel_prune=='channel':
            self.w=self.weight*self.adj*self.pre_adj
        else:
            self.adj=GetSubnet.apply(self.popup_scores.abs(), self.threshold)
            self.w = self.weight * self.adj
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
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False
        self.w = 0
        self.k=False
        self.threshold=False

    def set_threshold(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        # if self.threshold is False:
        #     adj=1.0
        # else:
        #     adj = GetSubnet.apply(self.popup_scores.abs(), self.threshold)
        adj=1.0
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)
        return x


class SubnetBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, batch_norm):
        super(SubnetBatchNorm2d, self).__init__(batch_norm.num_features)
        # Duplicate the weight parameter
        self.weight.data = batch_norm.weight.data.clone().detach()
        self.bias.data = batch_norm.bias.data.clone().detach()
        self.running_mean = batch_norm.running_mean.clone().detach()
        self.running_var = batch_norm.running_var.clone().detach()
        self.weight_mask = torch.ones_like(self.weight.data)
        
    def set_mask(self,adj):
        self.weight_mask=adj

    def forward(self, x):       
        result = super(SubnetBatchNorm2d, self).forward(x)
        if type(self.weight_mask) is float:
            mask=self.weight_mask
        else:
            mask=self.weight_mask.view(1,result.shape[1],1,1)
        result1=result*mask
        return result1


def replace_layers(model, old_layer, new_layer, channelPrune, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_layers(module, old_layer, new_layer,channelPrune,args)

        if type(module) == old_layer:
            if isinstance(module, nn.Conv2d):
                if args.network == 'vgg':
                    layer_new = new_layer(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, False,channelPrune)
                else:
                    layer_new = new_layer(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias,channelPrune)
            elif isinstance(module, nn.Linear):
                layer_new = new_layer(module.in_features, module.out_features, module.bias,channelPrune)
            elif isinstance(module, nn.BatchNorm2d):
                layer_new = new_layer(module)
            layer_new.weight.data=module.weight.data
            if args.network == 'vgg' and isinstance(module, nn.Conv2d):
                pass
            elif module.bias is not None:
                layer_new.bias.data=module.bias.data
            model._modules[name] = layer_new

    return model


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
        return SubnetConv, SubnetLinear, SubnetBatchNorm2d
    else:
        raise ValueError("Incorrect layer type")


def calculate_channel_sparsity(module):
    if isinstance(module,SubnetConv):
        if isinstance(module.adj, float):
            channel_mask_num = 0
            channel_num = module.weight.shape[0]
            channel_sparsity = 0
        else:
            channel_num = module.weight.shape[0]
            channel_mask_num = channel_num - module.adj.sum().item()
            channel_sparsity = channel_mask_num / channel_num
    else:
        channel_mask_num = 0
        channel_num = 0
        channel_sparsity = 0
    return channel_num, channel_mask_num, channel_sparsity


def calculate_flops(model):
    pass



def set_scored_network(network, args):
    cl, ll, bl = get_layers('subnet')
    network=replace_layers(network,Conv2d,cl,'channel',args)
    # network=replace_layers(network,Linear,ll,'channel',args)
    network=replace_layers(network,BatchNorm2d,bl,'channel',args)
    for name, weight in network.named_parameters():
        if hasattr(weight, 'is_score') and weight.is_score:
            print(name, weight.shape)
    network.to(args.device)
    return network


def set_prune_threshold(network, prune_rate):
    score_dict={}
    for name, weight in network.named_parameters():
        if 'conv' in name and hasattr(weight, 'is_score') and weight.is_score:
            score_dict[name] = torch.clone(weight.data).detach().abs_()
    global_scores = torch.cat([torch.flatten(v) for v in score_dict.values()])
    k = int((1 - prune_rate) * global_scores.numel())
    if not k < 1:
        threshold, _ = torch.kthvalue(global_scores, k)
        cl, ll, bl = get_layers('subnet')
        for name, module in network.named_modules():
            if isinstance(module,cl) and ('conv' in name):
                module.set_threshold(threshold)
            # if  isinstance(module,ll):
            #     module.set_threshold(threshold)


def set_channel_threshold(module, args):
    if isinstance(module, SubnetConv):
        scores = torch.flatten(torch.clone(module.popup_scores.data).detach().abs_())
    k = int(args.channel_max * scores.numel())
    threshold, _ = torch.kthvalue(scores, k)
    module.set_threshold(threshold)


def set_limited_threshold(model, args):
    normal=True
    set_prune_threshold(model, args.density)
    # print('before limit')
    # display_sparsity(model, args)
    conv1_t_c, conv1_m_c, conv1_s = calculate_channel_sparsity(model.conv1)
    conv2_t_c, conv2_m_c, conv2_s = calculate_channel_sparsity(model.layer1[0].conv1)
    conv3_t_c, conv3_m_c, conv3_s = calculate_channel_sparsity(model.layer1[0].conv2)
    conv4_t_c, conv4_m_c, conv4_s = calculate_channel_sparsity(model.layer1[1].conv1)
    conv5_t_c, conv5_m_c, conv5_s = calculate_channel_sparsity(model.layer1[1].conv2)
    if conv1_s > args.channel_max:
        normal=False
        set_channel_threshold(model.conv1, args)
        conv1_t_c, conv1_m_c, conv1_s = calculate_channel_sparsity(model.conv1)
    if conv2_s > args.channel_max:
        normal=False
        set_channel_threshold(model.layer1[0].conv1, args)
        conv2_t_c, conv2_m_c, conv2_s = calculate_channel_sparsity(model.layer1[0].conv1)
    if conv3_s > args.channel_max:
        normal=False
        set_channel_threshold(model.layer1[0].conv2, args)
        conv3_t_c, conv3_m_c, conv3_s = calculate_channel_sparsity(model.layer1[0].conv2)
    if conv4_s > args.channel_max:
        normal=False
        set_channel_threshold(model.layer1[1].conv1, args)
        conv4_t_c, conv4_m_c, conv4_s = calculate_channel_sparsity(model.layer1[1].conv1)
    if conv5_s > args.channel_max:
        normal=False
        set_channel_threshold(model.layer1[1].conv2, args)
        conv5_t_c, conv5_m_c, conv5_s = calculate_channel_sparsity(model.layer1[1].conv2)
    if not normal:
        limited_t_c = conv1_t_c + conv2_t_c + conv3_t_c + conv4_t_c + conv5_t_c
        limited_m_c = conv1_m_c + conv2_m_c + conv3_m_c + conv4_m_c + conv5_m_c
        score_dict={}
        for name, module in model.named_modules():
            if (name not in ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2']) and ('conv' in name) and isinstance(module, SubnetConv):
                score_dict[name] = torch.clone(module.popup_scores).detach().abs_()
        global_scores = torch.cat([torch.flatten(v) for v in score_dict.values()])
        k = int((1-args.density) * (global_scores.numel()+limited_t_c) - limited_m_c)
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for name, module in model.named_modules():
                if (name not in ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2']) and ('conv' in name) and isinstance(module, SubnetConv):
                    module.set_threshold(threshold)
        # print('after limit')
        # display_sparsity(model, args)



def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def switch_to_prune(model):
    # print(f"#################### Pruning network ####################")
    # print(f"===>>  gradient for weights: None  | training importance scores only")

    unfreeze_vars(model, "popup_scores")
    freeze_vars(model, "weight")
    freeze_vars(model, "bias")
    for param in model.fc.parameters():
        param.requires_grad= True


def switch_to_finetune(model):
    # print(f"#################### Fine-tuning network ####################")
    # print(
    #     f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
    # )
    freeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")
    cl, ll, bl = get_layers('subnet')
    for name, module in model.named_modules():
        if isinstance(module,cl):
            if isinstance(module.adj, torch.Tensor):
                module.adj = module.adj.detach()
            if isinstance(module.pre_adj, torch.Tensor):
                module.pre_adj = module.pre_adj.detach()


def switch_to_bilevel(model):
    unfreeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")


def Calculate_mask(model, args, bn_detach=True):
    set_limited_threshold(model, args)
    pre_adj=1.0
    pre_scores=1.0
    cur_adj=model.conv1.calculate_mask(pre_adj,pre_scores)
    cur_scores=model.conv1.popup_scores
    if bn_detach and type(cur_adj) is not float:
        model.bn1.set_mask(cur_adj.detach())
    else:
        model.bn1.set_mask(cur_adj)
    pre_adj=cur_adj
    pre_scores=cur_scores
    for name,module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):           
            if module.downsample is not None:
                if type(pre_adj) is float:
                    copy_adj=pre_adj
                    copy_scores=pre_scores
                else:
                    copy_adj=pre_adj.clone()
                    copy_scores=pre_scores.clone()
                # module.conv1.threshold=thres
                cur_adj=module.conv1.calculate_mask(pre_adj,pre_scores)
                cur_scores=module.conv1.popup_scores
                #print("name",name,thres)
                if bn_detach and type(cur_adj) is not float:
                    module.bn1.set_mask(cur_adj.detach())
                else:
                    module.bn1.set_mask(cur_adj)
                pre_adj=cur_adj
                pre_scores=cur_scores
                # module.conv2.threshold=thres
                cur_adj=module.conv2.calculate_mask(pre_adj,pre_scores)
                cur_scores=module.conv2.popup_scores
                
                if bn_detach and type(cur_adj) is not float:
                    module.bn2.set_mask(cur_adj.detach())
                else:
                    module.bn2.set_mask(cur_adj)


                pre_adj=cur_adj
                pre_scores=cur_scores
                # module.downsample[0].threshold=thres
                cur_adj=module.downsample[0].calculate_mask(copy_adj,1.0)
                cur_scores=module.downsample[0].popup_scores

                # if args.shortcut=='depend':
                module.downsample[0].adj=pre_adj
                # module.downsample[0].popup_scores=1.0

                #module.downsample[1].set_mask(pre_adj)
                if bn_detach and type(pre_adj) is not float:
                    module.downsample[1].set_mask(pre_adj.detach())
                else:
                    module.downsample[1].set_mask(pre_adj)
                # elif args.shortcut=='intersect':
                #     adj=((cur_adj+pre_adj)>0).long()
                #     module.conv2.adj=adj
                #     if bn_detach and type(adj) is not float:
                #         module.downsample[1].set_mask(adj.detach())
                #     else:
                #         module.downsample[1].set_mask(adj)
                #     module.downsample[0].adj=adj
                #     #module.downsample[1].set_mask(adj)
                #     pre_adj=adj
                    

                ##no need for bn2
            else:
                #copy_adj=pre_adj.clone()
                # module.conv1.threshold=thres
                cur_adj=module.conv1.calculate_mask(pre_adj,pre_scores)
                cur_scores=module.conv1.popup_scores
                #module.bn1.set_mask(cur_adj)
                if bn_detach and type(cur_adj) is not float:
                    module.bn1.set_mask(cur_adj.detach())
                else:
                    module.bn1.set_mask(cur_adj)
                pre_adj=cur_adj
                pre_scores=cur_scores

                # module.conv2.threshold=thres
                cur_adj=module.conv2.calculate_mask(pre_adj,pre_scores)
                cur_scores=module.conv2.popup_scores
                #module.bn2.set_mask(cur_adj)
                if bn_detach and type(cur_adj) is not float:
                    module.bn2.set_mask(cur_adj.detach())
                else:
                    module.bn2.set_mask(cur_adj)
                pre_adj=cur_adj
                pre_scores=cur_scores


def display_sparsity(network, args):
    print(f'Sparsity Level: {1-args.density}')
    print('Channel-wise Sparsity')
    cl, ll, bl = get_layers('subnet')
    total_channel_num = 0
    total_channel_mask_num = 0
    for name, module in network.named_modules():
        if isinstance(module,cl):
            if isinstance(module.adj, float):
                channel_mask_num = 0
                channel_num = module.weight.shape[0]
                channel_sparsity = 0
            else:
                channel_num = module.weight.shape[0]
                channel_mask_num = channel_num - module.adj.sum().item()
                channel_sparsity = channel_mask_num / channel_num
            total_channel_num += channel_num
            total_channel_mask_num += channel_mask_num
            print(f'name:{name}   score_shape:{module.popup_scores.shape}   channel_num:{channel_num}   channel_mask_num:{channel_mask_num}   channel_sparsity:{channel_sparsity}')
    print(f'Total channel-wise   total_channel_num:{total_channel_num}   total_channel_mask_num:{total_channel_mask_num}   channel_sparsity:{total_channel_mask_num / total_channel_num}')
    
    print('Parameter Sparsity')
    total_param_num = 0
    total_param_mask_num = 0
    for name, module in network.named_modules():
        if isinstance(module,cl):
            param_num = module.weight.numel()
            param_mask_num = param_num - torch.nonzero(module.weight*module.adj*module.pre_adj).size(0)
            param_sparsity = param_mask_num / param_num
            total_param_num += param_num
            total_param_mask_num += param_mask_num
            print(f'name: {name}   param_num:{param_num}   param_mask_num: {param_mask_num}   param_sparsity:{param_sparsity}')
    total_param_num += 512000
    print(f'Total param-wise   total_param_num: {total_param_num}   total_param_mask_num: {total_param_mask_num}   param_sparsity:{total_param_mask_num / total_param_num}')