import math

import math
import jittor as jt
import jittor.nn as nn
from jittor.init import kaiming_uniform_
import numpy as np

from PCR_Jittor.jittor.modules.kpconv.kernel_points import load_kernels
from PCR_Jittor.jittor.modules.ops import index_select
import time

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        # print("forloop size 2:", len(x.size()[n:]))
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return jt.exp(-sq_r / (2 * sig**2 + eps))

def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = jt.concat((x, jt.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features= jt.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # length.shape: [1]
        # Average features for each batch cloud
        averaged_features.append(jt.mean(x[i0:i0 + length.data[0]], dim=0))

        # Increment for next cloud
        i0 += length.data[0]

    # Average features in each batch
    return jt.stack(averaged_features)



class KPConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,radius, KP_extent,  p_dim=3,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()
        # print(kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
        #          fixed_kernel_points, KP_influence, aggregation_mode,
        #          deformable, modulated)
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = jt.zeros((self.K, in_channels, out_channels), dtype=jt.float32) # Parameter

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.in_channels,
                                      self.offset_dim,self.K,
                                      radius,
                                      KP_extent,
                                      p_dim=self.p_dim,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = jt.zeros(self.offset_dim, dtype=jt.float32) # Parameter

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()
        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return jt.array(K_points_numpy, dtype=jt.float32).stop_grad() # Parameter

    def execute(self,x, q_pts, s_pts, neighb_inds):
        # print("in: ", q_pts.shape, s_pts.shape, neighb_inds.shape, x.shape)
        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(x,q_pts, s_pts, neighb_inds) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * jt.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = jt.concat((s_pts, jt.zeros_like(s_pts[:1, :]) + 1e6), 0)
        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]
        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors = neighbors.unsqueeze(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = jt.sum(differences ** 2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = jt.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = jt.any(sq_distances < self.KP_extent ** 2, dim=2).type(jt.int32)

            # New value of max neighbors
            new_max_neighb = jt.max(jt.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = jt.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds = neighb_row_inds.unsqueeze(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(jt.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = jt.ones_like(sq_distances)
            all_weights = jt.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = jt.clamp(1 - jt.sqrt(sq_distances) / self.KP_extent, min_v=0.0)
            all_weights = jt.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = jt.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = jt.argmin(sq_distances, dim=2)
            all_weights *= jt.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = jt.concat((x, jt.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = jt.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = jt.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        out = jt.sum(kernel_outputs, dim=0)
        neighbor_features_sum = jt.sum(neighb_x, dim=-1)
        neighbor_num = jt.sum(jt.greater(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = jt.maximum(neighbor_num, jt.ones_like(neighbor_num))
        out = out / neighbor_num.unsqueeze(1)
        return out

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class KPConv_pure(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,radius, KP_extent,  p_dim=3,
                 fixed_kernel_points='center'):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv_pure, self).__init__()
        # print(kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
        #          fixed_kernel_points, KP_influence, aggregation_mode,
        #          deformable, modulated)
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points

        # Initialize weights
        self.weights = jt.zeros((self.K, in_channels, out_channels), dtype=jt.float32) # Parameter
        # Reset parameters
        self.reset_parameters()
        # Initialize kernel points
        self.kernel_points = self.init_KP()
        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)
        return jt.array(K_points_numpy, dtype=jt.float32).stop_grad() # Parameter

    def execute(self,x, q_pts, s_pts, neighb_inds):
        s_pts = jt.concat((s_pts, jt.zeros_like(s_pts[:1, :]) + 1e6), 0)
        neighbors =s_pts[neighb_inds, :]#index_select(s_pts, neighb_inds, dim=0)   #
        neighbors = neighbors - q_pts.unsqueeze(1)

        neighbors = neighbors.unsqueeze(2)
        differences = neighbors - self.kernel_points
        
        sq_distances = jt.sum(differences ** 2, dim=3)
        new_neighb_inds = neighb_inds
        all_weights = jt.clamp(1 - jt.sqrt(sq_distances) / self.KP_extent, min_v=0.0)
        all_weights = jt.transpose(all_weights, 1, 2)
        

        x = jt.concat((x, jt.zeros_like(x[:1, :])), 0)
        neighb_x = gather(x, new_neighb_inds) #index_select(x, new_neighb_inds, dim=0)  ##
        weighted_features = jt.matmul(all_weights, neighb_x)


        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = jt.matmul(weighted_features, self.weights)
        out = jt.sum(kernel_outputs, dim=0)


        neighbor_features_sum = jt.sum(neighb_x, dim=-1)
        neighbor_num = jt.sum(jt.greater(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = jt.maximum(neighbor_num, jt.ones_like(neighbor_num))
        out = out / neighbor_num.unsqueeze(1)

        return out

    def __repr__(self):
        return 'KPConv_pure(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)