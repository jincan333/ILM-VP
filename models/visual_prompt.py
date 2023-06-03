import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, args, normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        output_size = args.output_size
        input_size = args.input_size
        mask = torch.zeros(3, input_size, input_size)
        
        self.l_pad = int((output_size-input_size+1)/2)
        self.r_pad = int((output_size-input_size)/2)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))
        self.program = torch.nn.Parameter(data=torch.zeros(3, output_size, output_size))
        self.normalize = normalize

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x


class PadVisualPrompt(nn.Module):
    def __init__(self, args, normalize=None):
        super(PadVisualPrompt, self).__init__()
        pad = args.pad_size
        output_size = args.output_size
        input_size = args.input_size
        self.l_pad = int((output_size-input_size+1)/2)
        self.r_pad = int((output_size-input_size)/2)
        self.normalize=normalize
        self.program = torch.nn.Parameter(data=torch.zeros(3, output_size, output_size)) 

        if output_size > 2*pad:
            mask = torch.zeros(3, output_size-2*pad, output_size-2*pad)
            self.register_buffer("mask", F.pad(mask, [pad for _ in range(4)], value=1))
        elif output_size == 2*pad:
            mask = torch.ones(3, output_size, output_size)
            self.register_buffer("mask", mask)
        else:
            raise ValueError("Pad Should Not Exceed Half Of Output Size")

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x


class FixVisualPrompt(nn.Module):
    def __init__(self, args, normalize):
        super(FixVisualPrompt, self).__init__()
        mask_size = args.mask_size
        output_size = args.output_size
        input_size = args.input_size
        self.l_pad = int((output_size-input_size+1)/2)
        self.r_pad = int((output_size-input_size)/2)
        mask = torch.zeros(3, output_size, output_size)
        mask[:, :mask_size, :mask_size] = 1
    
        self.register_buffer("mask", mask)
        self.program = torch.nn.Parameter(data=torch.zeros(3, output_size, output_size))
        self.normalize = normalize

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x


class RandomVisualPrompt(nn.Module):
    def __init__(self, args, normalize):
        super(RandomVisualPrompt, self).__init__()
        mask_size = args.mask_size
        output_size = args.output_size
        input_size = args.input_size
        self.l_pad = int((output_size-input_size+1)/2)
        self.r_pad = int((output_size-input_size)/2)
        mask = torch.zeros(3, output_size, output_size)
        x_ = np.random.choice(output_size - mask_size)
        y_ = np.random.choice(output_size - mask_size)
        mask[:, x_ : x_ + mask_size, y_ : y_ + mask_size] = 1
    
        self.register_buffer("mask", mask)
        self.program = torch.nn.Parameter(data=torch.zeros(3, output_size, output_size))
        self.normalize = normalize

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x


