import torch
import torch.nn as nn
import unittest
import torch.nn.functional as F
import numpy as np

def print_inf_time(model,input_res=224):

    dummy_input = torch.randn(1, 3, input_res, input_res, dtype=torch.float).cuda()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 140
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(20):
        # print(f'warm up iteration {_}')
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            # print(f'forward pass iteration {rep}/{repetitions}')
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
    # print ("curr_time",curr_time)
            timings[rep] = curr_time
            
    repetitions=repetitions-20
    timings=[i[0] for i in timings]
    timings=np.array(timings)
    print (timings)
    print (np.argsort(timings))
    timings=timings[np.argsort(timings)[20:repetitions]]

    print ("=====0")
    print (timings)
    mean_syn = np.mean(timings) 
    std_syn = np.std(timings)
    print("mean_syn",np.round(mean_syn,2),np.round(std_syn,2))

    return np.round(mean_syn,2)



def print_throuput(model=None,input_res=224):
    optimal_batch_size=10
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    dummy_input = torch.randn(optimal_batch_size, 3,input_res,input_res, dtype=torch.float).to(device)
    repetitions=140
    total_time = []
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time.append(optimal_batch_size/curr_time)
   
     
    repetitions=repetitions-20
    total_time=np.array(total_time)
    print (total_time)
    print (np.argsort(total_time))
    total_time=total_time[np.argsort(total_time)[20:repetitions]]
    print ("====")
    print (total_time)          
    
    # Throughput =   ((repetitions-20)*optimal_batch_size)/np.sum(total_time)
    print ("Throughput",np.round(np.mean(total_time),2),np.round(np.std(total_time),2))
    return np.round(np.mean(total_time),2)



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1_0 = BasicBlock(64, 64)
        self.layer1_1 = BasicBlock(64, 64)

        self.layer2_0 = BasicBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))
        self.layer2_1 = BasicBlock(128, 128)

        self.layer3_0 = BasicBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ))
        self.layer3_1 = BasicBlock(256, 256)

        self.layer4_0 = BasicBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ))
        self.layer4_1 = BasicBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)

        x = self.layer2_0(x)
        x = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.layer4_1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet01(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet01, self).__init__()

        self.conv1 = nn.Conv2d(3, 55, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(55)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2_1_1 = nn.Conv2d(55, 58, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_1 = nn.BatchNorm2d(58)
        self.conv2_1_2 = nn.Conv2d(58, 55, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_2 = nn.BatchNorm2d(55)

        self.conv2_2_1 = nn.Conv2d(55, 51, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2_1 = nn.BatchNorm2d(51)
        self.conv2_2_2 = nn.Conv2d(51, 55, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2_2 = nn.BatchNorm2d(55)

        # Layer 2
        self.conv3_1_1 = nn.Conv2d(55, 99, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1_1 = nn.BatchNorm2d(99)
        self.conv3_1_2 = nn.Conv2d(99, 114, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1_2 = nn.BatchNorm2d(114)
        self.downsample3_1 = nn.Sequential(
            nn.Conv2d(55, 114, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(114)
        )

        self.conv3_2_1 = nn.Conv2d(114, 113, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2_1 = nn.BatchNorm2d(113)
        self.conv3_2_2 = nn.Conv2d(113, 114, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2_2 = nn.BatchNorm2d(114)

        # Layer 3
        self.conv4_1_1 = nn.Conv2d(114, 222, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1_1 = nn.BatchNorm2d(222)
        self.conv4_1_2 = nn.Conv2d(222, 234, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1_2 = nn.BatchNorm2d(234)
        self.downsample4_1 = nn.Sequential(
            nn.Conv2d(114, 234, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(234)
        )

        self.conv4_2_1 = nn.Conv2d(234, 230, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2_1 = nn.BatchNorm2d(230)
        self.conv4_2_2 = nn.Conv2d(230, 234, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2_2 = nn.BatchNorm2d(234)

        # Layer 4
        self.conv5_1_1 = nn.Conv2d(234, 468, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1_1 = nn.BatchNorm2d(468)
        self.conv5_1_2 = nn.Conv2d(468, 441, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1_2 = nn.BatchNorm2d(441)
        self.downsample5_1 = nn.Sequential(
            nn.Conv2d(234, 441, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(441)
        )

        self.conv5_2_1 = nn.Conv2d(441, 457, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2_1 = nn.BatchNorm2d(457)
        self.conv5_2_2 = nn.Conv2d(457, 441, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2_2 = nn.BatchNorm2d(441)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(441, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        residual = x
        x = self.conv2_1_1(x)
        x = self.bn2_1_1(x)
        x = self.relu(x)
        x = self.conv2_1_2(x)
        x = self.bn2_1_2(x)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv2_2_1(x)
        x = self.bn2_2_1(x)
        x = self.relu(x)
        x = self.conv2_2_2(x)
        x = self.bn2_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 2
        residual = x
        x = self.conv3_1_1(x)
        x = self.bn3_1_1(x)
        x = self.relu(x)
        x = self.conv3_1_2(x)
        x = self.bn3_1_2(x)
        residual = self.downsample3_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv3_2_1(x)
        x = self.bn3_2_1(x)
        x = self.relu(x)
        x = self.conv3_2_2(x)
        x = self.bn3_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 3
        residual = x
        x = self.conv4_1_1(x)
        x = self.bn4_1_1(x)
        x = self.relu(x)
        x = self.conv4_1_2(x)
        x = self.bn4_1_2(x)
        residual = self.downsample4_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv4_2_1(x)
        x = self.bn4_2_1(x)
        x = self.relu(x)
        x = self.conv4_2_2(x)
        x = self.bn4_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 4
        residual = x
        x = self.conv5_1_1(x)
        x = self.bn5_1_1(x)
        x = self.relu(x)
        x = self.conv5_1_2(x)
        x = self.bn5_1_2(x)
        residual = self.downsample5_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv5_2_1(x)
        x = self.bn5_2_1(x)
        x = self.relu(x)
        x = self.conv5_2_2(x)
        x = self.bn5_2_2(x)
        x += residual
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class ResNet02(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet02, self).__init__()

        self.conv1 = nn.Conv2d(3, 52, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(52)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2_1_1 = nn.Conv2d(52, 58, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_1 = nn.BatchNorm2d(58)
        self.conv2_1_2 = nn.Conv2d(58, 52, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_2 = nn.BatchNorm2d(52)

        self.conv2_2_1 = nn.Conv2d(52, 53, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2_1 = nn.BatchNorm2d(53)
        self.conv2_2_2 = nn.Conv2d(53, 52, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2_2 = nn.BatchNorm2d(52)

        # Layer 2
        self.conv3_1_1 = nn.Conv2d(52, 101, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1_1 = nn.BatchNorm2d(101)
        self.conv3_1_2 = nn.Conv2d(101, 109, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1_2 = nn.BatchNorm2d(109)
        self.downsample3_1 = nn.Sequential(
            nn.Conv2d(52, 109, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(109)
        )

        self.conv3_2_1 = nn.Conv2d(109, 108, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2_1 = nn.BatchNorm2d(108)
        self.conv3_2_2 = nn.Conv2d(108, 109, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2_2 = nn.BatchNorm2d(109)

        # Layer 3
        self.conv4_1_1 = nn.Conv2d(109, 212, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1_1 = nn.BatchNorm2d(212)
        self.conv4_1_2 = nn.Conv2d(212, 217, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1_2 = nn.BatchNorm2d(217)
        self.downsample4_1 = nn.Sequential(
            nn.Conv2d(109, 217, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(217)
        )

        self.conv4_2_1 = nn.Conv2d(217, 213, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2_1 = nn.BatchNorm2d(213)
        self.conv4_2_2 = nn.Conv2d(213, 217, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2_2 = nn.BatchNorm2d(217)

        # Layer 4
        self.conv5_1_1 = nn.Conv2d(217, 405, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1_1 = nn.BatchNorm2d(405)
        self.conv5_1_2 = nn.Conv2d(405, 345, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1_2 = nn.BatchNorm2d(345)
        self.downsample5_1 = nn.Sequential(
            nn.Conv2d(217, 345, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(345)
        )

        self.conv5_2_1 = nn.Conv2d(345, 354, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2_1 = nn.BatchNorm2d(354)
        self.conv5_2_2 = nn.Conv2d(354, 345, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2_2 = nn.BatchNorm2d(345)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(345, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        residual = x
        x = self.conv2_1_1(x)
        x = self.bn2_1_1(x)
        x = self.relu(x)
        x = self.conv2_1_2(x)
        x = self.bn2_1_2(x)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv2_2_1(x)
        x = self.bn2_2_1(x)
        x = self.relu(x)
        x = self.conv2_2_2(x)
        x = self.bn2_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 2
        residual = x
        x = self.conv3_1_1(x)
        x = self.bn3_1_1(x)
        x = self.relu(x)
        x = self.conv3_1_2(x)
        x = self.bn3_1_2(x)
        residual = self.downsample3_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv3_2_1(x)
        x = self.bn3_2_1(x)
        x = self.relu(x)
        x = self.conv3_2_2(x)
        x = self.bn3_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 3
        residual = x
        x = self.conv4_1_1(x)
        x = self.bn4_1_1(x)
        x = self.relu(x)
        x = self.conv4_1_2(x)
        x = self.bn4_1_2(x)
        residual = self.downsample4_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv4_2_1(x)
        x = self.bn4_2_1(x)
        x = self.relu(x)
        x = self.conv4_2_2(x)
        x = self.bn4_2_2(x)
        x += residual
        x = self.relu(x)

        # Layer 4
        residual = x
        x = self.conv5_1_1(x)
        x = self.bn5_1_1(x)
        x = self.relu(x)
        x = self.conv5_1_2(x)
        x = self.bn5_1_2(x)
        residual = self.downsample5_1(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.conv5_2_1(x)
        x = self.bn5_2_1(x)
        x = self.relu(x)
        x = self.conv5_2_2(x)
        x = self.bn5_2_2(x)
        x += residual
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Instantiate the ResNet-18 model
print('*'*100)
print('resnet18')
resnet18 = ResNet(num_classes=1000).cuda()
print(resnet18)
print_inf_time(resnet18)
print_throuput(resnet18)
print('*'*100)

# Instantiate the ResNet-18 model
print('*'*100)
print('resnet18_01')
resnet18_01 = ResNet01(num_classes=1000).cuda()
print(resnet18_01)
print_inf_time(resnet18_01)
print_throuput(resnet18_01)
print('*'*100)

# Instantiate the ResNet-18 model
print('*'*100)
print('resnet18_02')
resnet18_02 = ResNet02(num_classes=1000).cuda()
print(resnet18_02)
print_inf_time(resnet18_02)
print_throuput(resnet18_02)
print('*'*100)