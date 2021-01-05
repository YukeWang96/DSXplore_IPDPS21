#!/usr/bin/env python3
import torch
import torch.nn as nn
import scc_cuda

# X = torch.rand((6,4,1,1)).cuda()
# w = torch.rand((7, 2)).cuda()
# out = torch.zeros((6,7,1,1)).cuda()
# # X.cuda()
# # w.cuda()
# overlap = 0.5
# output = scc_cuda.forward(X, w, out, overlap)
# print(output[0])
# print(output[0].size())

# d_out = torch.ones((6,7,1,1)).cuda()
# X = torch.ones((6,4,1,1)).cuda()
# w = torch.ones((7,2)).cuda()
# d_X = torch.zeros_like(X).cuda()
# d_w = torch.zeros_like(w).cuda()
# overlap = 0.5
# d_output = scc_cuda.backward(d_out, X, w, d_X, d_w, overlap)
# print("d_X: ", d_output[0])
# print("d_w: ", d_output[1])

overlap = 0.5
X_in = torch.ones((6,4,1,1)).cuda()
X_out = torch.zeros((6,2*7,1,1)).cuda()
output = scc_cuda.forward(X_in, X_out, 2, overlap)[0]
print(output.size())


d_Xout = torch.ones((6,2*7,1,1)).cuda()
d_Xin = torch.zeros((6,4,1,1)).cuda()
output = scc_cuda.forward(X_in, X_out, 2, overlap)
input_tensor = scc_cuda.forward(d_Xout, d_Xin, 2, overlap)[0]
print(input_tensor.size())

'''
class RPWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, output, bias, overlap):
        ctx.overlap = overlap
        output = scc_cuda.forward(input, weights, output, overlap)[0]
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weights = ctx.saved_tensors
        d_input = torch.zeros_like(input).cuda()
        d_weights = torch.zeros_like(weights).cuda()
        d_input, d_weights = scc_cuda.backward(d_output, input, weights, d_input, d_weights, ctx.overlap)
        return d_input, d_weights, d_output, None, None

class RPW(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grp_n, overlap):
        super(RPW, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.grp_n = grp_n
        self.overlap = overlap
        
        # print(self.input_channel, self.grp_n)
        assert self.input_channel % self.grp_n == 0
        self.input_grp_unit_dim = int(self.input_channel/self.grp_n)
        self.weights = torch.nn.Parameter(torch.zeros(self.output_channel, self.input_grp_unit_dim))
        self.bias = torch.nn.Parameter(torch.ones(self.output_channel))
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # stdv = 1.0 / math.sqrt(self.state_size)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        batch_size = input.size(0)
        h = input.size(2)
        w = input.size(3)
        self.output_tmp = torch.zeros((batch_size, self.output_channel, h, w)).cuda()
        return RPWFunction.apply(input, self.weights, self.output_tmp, self.bias, self.overlap)
'''