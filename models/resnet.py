'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scc_conv import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, channel_groups=2, overlap=0.5):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes)
        self.scc   = SCC(planes, planes, channel_groups, overlap)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.conv2(out)
        out = self.scc(out)
        out = self.bn2(out)
        # out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, channel_groups=2, overlap=0.5):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                    #    stride=stride, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, groups=planes)
        self.scc   = SCC(planes, planes, channel_groups, overlap)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.conv2(out)
        out = self.scc(out)
        out = F.relu(self.bn2(out))
        # out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, groups=2, oap=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.channel_groups = groups
        self.overlap = oap

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, channel_groups=self.channel_groups, overlap=self.overlap)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,channel_groups= self.channel_groups,overlap= self.overlap)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,channel_groups= self.channel_groups,overlap= self.overlap)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,channel_groups= self.channel_groups,overlap= self.overlap)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, channel_groups, overlap):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, channel_groups, overlap))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(groups=2, oap=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], groups=groups, oap=oap)


def ResNet34(groups=2, oap=0.5):
    return ResNet(BasicBlock, [3, 4, 6, 3], groups=groups, oap=oap)


def ResNet50(groups=2, oap=0.5):
    return ResNet(Bottleneck, [3, 4, 6, 3], groups=groups, oap=oap)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18().cuda()
    y = net(torch.randn(1, 3, 32, 32).cuda())
    print(y.size())

if __name__ == "__main__":
    test()