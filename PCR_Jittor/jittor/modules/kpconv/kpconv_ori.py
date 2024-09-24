import math

import jittor as jt
import jittor.nn as nn
from jittor.init import kaiming_uniform_
import numpy as np
from jittor.contrib import concat
from PCR_Jittor.jittor.modules.kpconv.kernel_points import load_kernels

from PCR_Jittor.jittor.modules.ops import index_select



class KPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        bias=False,
        dimension=3,
        inf=1e6,
        eps=1e-9,
    ):
        """Initialize parameters for KPConv.

        Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

        Deformable KPConv is not supported.

        Args:
             in_channels: dimension of input features.
             out_channels: dimension of output features.
             kernel_size: Number of kernel points.
             radius: radius used for kernel point init.
             sigma: influence radius of each kernel point.
             bias: use bias or not (default: False)
             dimension: dimension of the point space.
             inf: value of infinity to generate the padding point
             eps: epsilon for gaussian influence
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension

        self.inf = inf
        self.eps = eps

        # Initialize weights
        self.weights = nn.Parameter(jt.zeros(self.kernel_size, in_channels, out_channels))
        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.initialize_kernel_points()  # (N, 3)
        return
    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def initialize_kernel_points(self):
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed='center')
        return jt.array(kernel_points, dtype=jt.float32).stop_grad()

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        r"""KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        s_points =concat([s_points, jt.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)
        neighbors = index_select(s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = jt.sum(differences ** 2, dim=3)  # (M, H, K)
        neighbor_weights = jt.clamp(1 - jt.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = jt.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        s_feats = concat((s_feats, jt.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)  # (N+1, C) -> (M, H, C)
        weighted_feats = jt.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        weighted_feats = weighted_feats.permute((1, 0, 2))  # (M, K, C) -> (K, M, C)
        kernel_outputs = jt.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, C_out) -> (K, M, C_out)
        output_feats = jt.sum(kernel_outputs, dim=0)  # (K, M, C_out) -> (M, C_out)

        # normalization
        neighbor_feats_sum = jt.sum(neighbor_feats, dim=-1)
        neighbor_num = jt.sum(jt.greater(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = jt.maximum(neighbor_num, jt.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        """ if self.bias is not None:
            output_feats = output_feats + self.bias """

        return output_feats

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'kernel_size: {}'.format(self.kernel_size)
        format_string += ', in_channels: {}'.format(self.in_channels)
        format_string += ', out_channels: {}'.format(self.out_channels)
        format_string += ', radius: {:g}'.format(self.radius)
        format_string += ', sigma: {:g}'.format(self.sigma)
        format_string += ', bias: {}'.format(self.bias is not None)
        format_string += ')'
        return format_string
