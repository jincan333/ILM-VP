
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d
from torchvision import models
from torch.autograd import Variable
from torchvision.models import resnet18, ResNet18_Weights



def count_model_param_flops(model=None,channels=None, input_res=224, multiply_adds=True,s_list=None):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)

        num_weight_params = (self.weight.data != 0).float().sum()
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        num_weight_params = (self.weight.data != 0).float().sum()
        weight_ops = num_weight_params * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            # if isinstance(net, torch.nn.BatchNorm2d):
            #     net.register_forward_hook(bn_hook)
            # if isinstance(net, torch.nn.ReLU):
            #     net.register_forward_hook(relu_hook)
            # if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            #     net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                 net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    # if model==None:
    #     model = models.resnet18().cuda()

    # num_channel=model.conv1.weight.shape[0]
    # i=0
    # s=s_list[i]

    # model.conv1.weight.data[:int(num_channel*s)]=0
    # i+=1

    # pre_num_ch=int(num_channel*s)
    # for name,module in model.named_modules():
    #   if isinstance(module, models.resnet.BasicBlock):
    #       #module.conv1=create_conv(module.conv1,new_layer,channelPrune,GenerateMask,Samplingnet)
    #       copy_ch=pre_num_ch
    #       print(copy_ch)
    #       num_channel=module.conv1.weight.shape[0]
    #       s=s_list[i]
    #       module.conv1.weight.data[:int(num_channel*s)]=0
    #       module.conv1.weight.data[:,:pre_num_ch,:,:]=0
    #       pre_num_ch=int(num_channel*s)
    #       i+=1
    #       s=s_list[i]
    #       num_channel=module.conv2.weight.shape[0]
    #       module.conv2.weight.data[:int(num_channel*s)]=0
    #       module.conv2.weight.data[:,:pre_num_ch,:,:]=0
    #       pre_num_ch=int(num_channel*s)
    #       i+=1

    #       if module.downsample is not None:
    #           #module.downsample[0]=create_conv(module.downsample[0],new_layer,channelPrune,GenerateMask,Samplingnet)
    #           #if args.shortcut=='none':
    #           #    module.downsample[0].is_train=False
    #           #    module.conv2.is_train=False
    #           #elif  args.shortcut=='depend':
    #           num_channel=module.downsample[0].weight.shape[0]
    #           module.downsample[0].weight.data[:int(num_channel*s)]
    #           module.downsample[0].weight.data[:,:copy_ch,:,:]=0

    #               #module.conv2.is_train=True
    #           #module.downsample[1]=PrunableBatchNorm2d(module.downsample[1])
    #   elif  isinstance(module, models.resnet.Bottleneck):
    #       copy_ch=pre_num_ch
    #       num_channel=module.conv1.weight.shape[0]
    #       module.conv1.weight.data[:int(num_channel*s)]=0
    #       module.conv1.weight.data[:,:pre_num_ch,:,:]=0
    #       pre_num_ch=int(num_channel*s)
    #       num_channel=module.conv2.weight.shape[0]
    #       module.conv2.weight.data[:int(num_channel*s)]=0
    #       module.conv2.weight.data[:,:pre_num_ch,:,:]=0
    #       pre_num_ch=int(num_channel*s)
    #       num_channel=module.conv3.weight.shape[0]
    #       module.conv3.weight.data[:int(num_channel*s)]=0
    #       module.conv3.weight.data[:,:pre_num_ch,:,:]=0
    #       pre_num_ch=int(num_channel*s)
    #       if module.downsample is not None:
    #           num_channel=module.downsample[0].weight.shape[0]
    #           module.downsample[0].weight.data[:int(num_channel*s)]=0
    #           module.downsample[0].weight.data[:,:copy_ch,:,:]=0
    #   elif isinstance(module,torch.nn.Linear):
    #     module.weight.data[:,:pre_num_ch]=0
    #     #print(pre_num_ch)

    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True).cuda()
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.2f' % (total_flops/2/1000000))

    return total_flops/2/1000000

density_list=[0.9, 0.8, 0.7, 0.6, 0.5]
c_sparsity_list=[
    [0.125, 0, 0, 0, 0, 0.016, 0.016, 0.008, 0, 0.008, 0, 0.004, 0, 0.010, 0.004, 0.014, 0.704],
    [0.172, 0.079, 0, 0.094, 0, 0.141, 0.094, 0.047, 0.024, 0.079, 0.094, 0.098, 0.008, 0.125, 0.1, 0.135, 0.907],
    [0.21875, 0.125, 0.03125, 0.171875, 0.015625, 0.1796875, 0.234375, 0.171875, 0.1015625, 0.17578125, 0.2109375, 0.25, 0.06640625, 0.24609375, 0.24609375, 0.259765625, 0.94140625],
    [0.21875, 0.140625, 0.015625, 0.0625, 0, 0.171875, 0.125, 0.125, 0.015625, 0.15234375, 0.16796875, 0.19921875, 0.0390625, 0.267578125, 0.37890625, 0.9609375, 0.998046875],
    [0.234375, 0.109375, 0.046875, 0.125, 0, 0.1015625, 0.1875, 0.2109375, 0.0625, 0.296875, 0.2734375, 0.2734375, 0.07421875, 0.37109375, 0.77734375, 0.998046875, 0.998046875]
         ]
network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
for d, s_list in zip(density_list, c_sparsity_list):
    flops=count_model_param_flops(model=network, s_list=s_list)
    print(f'channel density: {d:.2f}, layer_cnt: {len(s_list): 2d}, flops: {flops:.2f}, speedup: {1823/flops: .2f}')