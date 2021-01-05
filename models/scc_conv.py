#!/usr/bin/env python3
import torch
import torch.nn as nn
import scc_cuda
import math
import time

# X = torch.rand((6,4,1,1)).cuda()
# w = torch.rand((7, 2)).cuda()
# out = torch.zeros((6,7,1,1)).cuda()
# # X.cuda()
# # w.cuda()
# overlap = 0.5
# output = scc_cuda.forward(X, w, out, overlap)
# print(output[0])
# print(output[0].size())

# d_out = torch.rand((6,7,1,1)).cuda()
# X = torch.rand((6,4,1,1)).cuda()
# w = torch.rand((7,2)).cuda()
# d_X = torch.zeros_like(X).cuda()
# d_w = torch.zeros_like(w).cuda()
# overlap = 0.5
# d_output = scc_cuda.backward(d_out, X, w, d_X, d_w, overlap)
# print("d_X: ", d_output[0])
# print("d_w: ", d_output[1])

# from torch.utils.cpp_extension import load
# scc_conv = load(name='DW_scc', sources=['scc_cuda.cpp', 'scc_cuda_kernel.cu'])
# print(scc_conv)
'''
class sccFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output, unit_dim, overlap):
        ctx.overlap = overlap
        ctx.unit_dim = unit_dim
        ctx.input_shape = input.size()
        # print("hello")
        output = scc_cuda.forward(input, output, unit_dim, overlap)[0]
        return output

    @staticmethod
    def backward(ctx, d_output):
        d_input = torch.zeros(ctx.input_shape).cuda()
        d_input = scc_cuda.backward(d_output, d_input, ctx.unit_dim, ctx.overlap)[0]
        return d_input, None, None, None

class scc(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grp_n, overlap):
        super(scc, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.grp_n = grp_n
        self.overlap = overlap
        
        assert self.input_channel % self.grp_n == 0
        self.unit_dim = int(self.input_channel//self.grp_n)
        self.conv2D = nn.Conv2d(self.output_channel * self.unit_dim, self.output_channel, kernel_size=1, groups=self.output_channel)

    def forward(self, input):
        batch_size = input.size(0)
        h = input.size(2)
        w = input.size(3)
        self.output_tmp = torch.zeros((batch_size, self.output_channel * self.unit_dim, h, w)).cuda()
        scc_output = sccFunction.apply(input, self.output_tmp, self.unit_dim, self.overlap)
        return self.conv2D(scc_output)
'''
class SCCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, output, overlap):
        ctx.overlap = overlap
        # start = time.perf_counter()
        output = scc_cuda.forward(input, weights, output, overlap)[0]
        ctx.save_for_backward(input, weights)
        # dur = time.perf_counter() - start
        # print("forward: {}s".format(dur))
        # print('forward')
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weights = ctx.saved_tensors
        d_input = torch.zeros(input.size()).cuda()
        d_weights = torch.zeros(weights.size()).cuda()
        # start = time.perf_counter()
        d_input, d_weights = scc_cuda.backward(d_output, input, weights, d_input, d_weights, ctx.overlap)
        # dur = time.perf_counter() - start
        # print("backward: {}s".format(dur))
        # d_input = torch.div(d_input, d_input_cnt)
        # d_weights = torch.div(d_weights, d_weights_cnt)
        # print(d_weights)
        # print("backward")
        return d_input, d_weights, None, None, None

class SCC(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grp_n, overlap):
        super(SCC, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.grp_n = grp_n
        self.overlap = overlap
        
        # print(self.input_channel, self.grp_n)
        assert self.input_channel % self.grp_n == 0
        self.input_grp_unit_dim = int(self.input_channel/self.grp_n)
        self.weights = torch.nn.Parameter(torch.randn(self.output_channel, self.input_grp_unit_dim))
        self.bias = torch.nn.Parameter(torch.ones(self.output_channel))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        h = input.size(2)
        w = input.size(3)
        self.output_tmp = torch.zeros((batch_size, self.output_channel, h, w)).cuda()
        # bias = self.bias.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).unsqueeze(3)
        scc_output = SCCFunction.apply(input, self.weights, self.output_tmp, self.overlap)
        # return scc_output + bias
        return scc_output