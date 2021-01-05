#!/usr/bin/env python3
import torch
import torch.nn as nn
import scc_cuda
import math
import time

class SCCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, output, overlap):
        ctx.overlap = overlap
        output = scc_cuda.forward(input, weights, output, overlap)[0]
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weights = ctx.saved_tensors
        d_input = torch.zeros(input.size()).cuda()
        d_weights = torch.zeros(weights.size()).cuda()
        d_input, d_weights = scc_cuda.backward(d_output, input, weights, d_input, d_weights, ctx.overlap)
        return d_input, d_weights, None, None, None

class SCC(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grp_n, overlap):
        super(SCC, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.grp_n = grp_n
        self.overlap = overlap
        
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