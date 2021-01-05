import torch
import torch.nn as nn
import sys

class RPW(nn.Module):
    def __init__(self, input_channel, output_channel, num_groups, overlap):
        super(RPW, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        assert input_channel % num_groups == 0
        assert output_channel % num_groups == 0

        # pre-compute slice point [start, end)
        self.slice_li = []
        self.width = int(input_channel/num_groups)
        start, end = 0, self.width
        start_v, end_v = start, end
        self.item_set = set()
        for fid in range(output_channel):
            item  = (start, end)
            if item not in self.item_set:
                self.item_set.add(item)
            else:
                break

            self.slice_li.append(item)
            start_v = end_v - int(overlap * self.width) 
            end_v = start_v + self.width

            start = start_v % self.input_channel
            end = end_v % self.input_channel

        self.conv2D = nn.Conv2d(self.width * len(self.item_set), output_channel, kernel_size=1, groups=len(self.item_set))

    def forward(self, input):
        combined_unit = []
        for idx in range(len(self.item_set)):
            item = self.slice_li[idx]
            start, end = item[0], item[1]
            if start > end and start < self.input_channel:
                tmp = input[:, start:, :, :]
                tmp_1 = input[:, :end, :, :]
                new_tmp = torch.cat([tmp, tmp_1], dim=1)
                combined_unit.append(new_tmp)
            else:
                combined_unit.append(input[:, start:end, :, :])
        
        combined_tensor = torch.cat(combined_unit, dim=1)
        results = self.conv2D(combined_tensor)
        return results


def DW_GPW(input_channel, output_channel, kernel_size, padding, num_groups):
    '''
    build the DW_GPW kernel
    '''
    DW_conv = nn.Conv2d(input_channel, input_channel, kernel_size, padding = 1, groups=input_channel)
    if input_channel % num_groups != 0:
        GPW_conv = nn.Conv2d(input_channel, output_channel, kernel_size=1, groups=1)
    else:
        GPW_conv = nn.Conv2d(input_channel, output_channel, kernel_size=1, groups=num_groups)
        
    return [DW_conv, GPW_conv]


def DW_RPW(input_channel, output_channel, kernel_size, padding, num_groups, overlap):
    '''
    build the DW_RPW kernel
    '''
    # Depth-Wise Convolution (DW)
    DW_conv = nn.Conv2d(input_channel, input_channel, kernel_size, padding=1, groups=input_channel)

    # Rolling Point-Wise (RPW) Convolution
    assert input_channel % num_groups == 0
    RPW_conv = RPW(input_channel, output_channel, num_groups=num_groups, overlap=overlap)

    return [DW_conv, RPW_conv]