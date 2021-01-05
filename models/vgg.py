'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

## only one of the following two package should be imported
from .scc_conv import *    # CUDA-based SCC implementation.
# from .DW_SCC import *    # Pytorch-based DW+PW, DW+GPW, SCC implementation.

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channel_groups=2, overlap=0.5, origin=False):
        super(VGG, self).__init__()
        self.channel_groups = channel_groups
        self.overlap = overlap
        self.origin = False
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.origin:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                else:
                    if in_channels == 3: 
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                    else:
                        # Only set one of the following four options while keep other commented out.

                        ## 1. Standard Convolution 
                        # layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]

                        ## 2. Pytorch-based DW+GPW, 
                        # layers += DW_SCC(in_channels, x, kernel_size=3, padding=1, num_groups=channel_groups, overlap=overlap)
                        
                        ## 3. Pytorch-based SCC implementation.
                        # layers += DW_GPW(in_channels, x, kernel_size=3, padding=1, num_groups=channel_groups)
                        
                        ## 4. CUDA-based SCC implementation.
                        layers += [
                                    torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), \
                                    SCC(in_channels, x, self.channel_groups, self.overlap)
                                    ]
                layers += [nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
