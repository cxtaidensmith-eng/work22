import math

import torch
from torch import nn


class ChebGraphConv(nn.Module):
    """K-order Chebyshev graph convolution."""

    def __init__(self, K, in_features, out_features, bias=True):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        if self.K < 0:
            raise ValueError('Chebyshev order must be non-negative')
        features = [x]
        if self.K >= 1:
            multiply = torch.sparse.mm if gso.is_sparse else torch.mm
            if not gso.is_sparse and x.is_sparse:
                x = x.to_dense()
                features[0] = x
            features.append(multiply(gso, x))
            for order in range(2, self.K):
                features.append(multiply(2 * gso, features[order - 1]) - features[order - 2])

        stacked = torch.stack(features, dim=0)
        if stacked.is_sparse:
            stacked = stacked.to_dense()
        output = torch.einsum('bij,bjk->ik', stacked, self.weight)
        return output if self.bias is None else output + self.bias

    def extra_repr(self):
        return f'K={self.K}, bias={self.bias is not None}'
