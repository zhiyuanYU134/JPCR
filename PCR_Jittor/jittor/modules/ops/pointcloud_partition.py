import warnings

import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
from scipy.spatial import cKDTree
from PCR_Jittor.jittor.modules.ops.pairwise_distance import pairwise_distance
from PCR_Jittor.jittor.modules.ops.index_select import index_select



def get_point_to_node_indices(points, nodes, return_counts = False):
    r"""Compute Point-to-Node partition indices of the point cloud.

    Distribute points to the nearest node. Each point is distributed to only one node.

    Args:
        points (Tensor): point cloud (N, C)
        nodes (Tensor): node set (M, C)
        return_counts (bool=False): whether return the number of points in each node.

    Returns:
        indices (LongTensor): index of the node that each point belongs to (N,)
        node_sizes (longTensor): the number of points in each node.
    """
    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.argmin(dim=1)[0]
    if return_counts:
        unique_indices, unique_counts = jt.unique(indices, return_counts=True)
        node_sizes = jt.zeros(nodes.shape[0], dtype='long')
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes
    else:
        return indices


@jt.no_grad()
def knn_partition(points, nodes, k, return_distance= False):
    r"""k-NN partition of the point cloud.

    Find the k nearest points for each node.

    Args:
        points: torch.Tensor (num_point, num_channel)
        nodes: torch.Tensor (num_node, num_channel)
        k: int
        return_distance: bool

    Returns:
        knn_indices: torch.Tensor (num_node, k)
        knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    sq_dist_mat = pairwise_distance(nodes, points)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k, largest=False)
    if return_distance:
        knn_distances = jt.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


@jt.no_grad()
def point_to_node_partition(
    points,
    nodes,
    point_limit
):
    r"""Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.argmin(dim=0)[0]  # (N,)
    node_masks = jt.zeros(nodes.shape[0],dtype='bool')# (M,)
    #node_masks.index_fill_(0, point_to_node, True)

    point_to_node_key=jt.unique(point_to_node)
    node_masks[point_to_node_key]=True

    matching_masks = jt.zeros_like(sq_dist_mat).bool()  # (M, N)
    point_indices = jt.arange(points.shape[0])# (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat=jt.masked_fill(sq_dist_mat,jt.logical_not(matching_masks), 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = jt.arange(nodes.shape[0]).unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = jt.equal(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices=jt.masked_fill(node_knn_indices,jt.logical_not(node_knn_masks), points.shape[0])
    return point_to_node, node_masks, node_knn_indices, node_knn_masks

