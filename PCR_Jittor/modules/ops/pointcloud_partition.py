import warnings

import torch
import numpy as np
from scipy.spatial import cKDTree
from PCR_Jittor.modules.ops.pairwise_distance import pairwise_distance
from PCR_Jittor.modules.ops.index_select import index_select
def pair_dist(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1. This enables us to use 2 instead of a2 + b2 for
        simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    if isinstance(points0, torch.Tensor):
        if len(points0.shape)==3:
            
            dist2 = (points0.unsqueeze(2)-points1.unsqueeze(1)).abs().pow(2).sum(-1)
        else:
            dist2 = (points0.unsqueeze(1)-points1.unsqueeze(0)).abs().pow(2).sum(-1)
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2
def get_nearest_neighbor(ref_points, src_points, return_index=False):
    r"""
    [PyTorch/Numpy] For each item in ref_points, find its nearest neighbor in src_points.

    The PyTorch implementation is based on pairwise distances, thus it cannot be used for large point clouds.
    """
    if isinstance(ref_points, torch.Tensor):
        distances = pair_dist(ref_points, src_points)
        nn_distances, nn_indices = distances.min(dim=1)
        if return_index:
            return nn_distances, nn_indices
        else:
            return nn_distances
    else:
        kd_tree1 = cKDTree(src_points)
        distances, indices = kd_tree1.query(ref_points, k=1, n_jobs=-1)
        if return_index:
            return distances, indices
        else:
            return distances


def get_point_to_node(points, nodes, return_counts=False):
    r"""
    [PyTorch/Numpy] Distribute points to the nearest node. Each point is distributed to only one node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param return_counts: bool (default: False)
        If True, return the number of points in each node.
    :return: indices: torch.Tensor (num_point)
        The indices of the nodes to which the points are distributed.
    """
    if isinstance(points, torch.Tensor):
        """ print("isinstance") """
        distances = pair_dist(points, nodes)
        indices = distances.min(dim=1)[1]
        if return_counts:
            unique_indices, unique_counts = torch.unique(indices, return_counts=True)
            node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices
    else:
        """ print("instance") """
        _, indices = get_nearest_neighbor(points, nodes, return_index=True)
        if return_counts:
            unique_indices, unique_counts = np.unique(indices, return_counts=True)
            node_sizes = np.zeros(nodes.shape[0], dtype=np.int64)
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices



def get_knn_indices(points, nodes, k, return_distance=False):
    r"""
    [PyTorch] Find the k nearest points for each node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param k: int
    :param return_distance: bool
    :return knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    dists = pair_dist(nodes, points)
    knn_distances, knn_indices = dists.topk(dim=1, k=k, largest=False)
    if return_distance:
        return torch.sqrt(knn_distances), knn_indices
    else:
        return knn_indices
@torch.no_grad()
def get_point_to_node_indices_and_masks(points, nodes, num_sample, return_counts=False):
    r"""
    [PyTorch] Perform point-to-node partition to the point cloud.

    :param points: torch.Tensor (num_point, 3)
    :param nodes: torch.Tensor (num_node, 3)
    :param num_sample: int
    :param return_counts: bool, whether to return `node_sizes`

    :return point_node_indices: torch.LongTensor (num_point,)
    :return node_sizes [Optional]: torch.LongTensor (num_node,)
    :return node_masks: torch.BoolTensor (num_node,)
    :return node_knn_indices: torch.LongTensor (num_node, max_point)
    :return node_knn_masks: torch.BoolTensor (num_node, max_point)
    """
    """ start_time=time.time() """

    point_to_node, node_sizes = get_point_to_node(points, nodes, return_counts=True)
    node_masks = torch.gt(node_sizes, 0)


    """ loading_time = time.time() - start_time
    print("get_point_to_node")
    print(loading_time) """

    node_knn_indices = get_knn_indices(points, nodes, num_sample)  # (num_node, max_point)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, num_sample)
    if len(points)<num_sample:
        knn_indices=torch.full([len(nodes),num_sample],nodes.shape[0]).cuda()
        knn_indices[:,:len(points)]=point_to_node[node_knn_indices]
    else:
        knn_indices=point_to_node[node_knn_indices]


    node_knn_masks = torch.eq(knn_indices, node_indices)
    sentinel_indices = torch.full_like(knn_indices, points.shape[0])
    node_knn_indices = torch.where(node_knn_masks, knn_indices, sentinel_indices)


    if return_counts:
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks

def get_point_to_node_indices(points: torch.Tensor, nodes: torch.Tensor, return_counts: bool = False):
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
    indices = sq_dist_mat.min(dim=1)[1]
    if return_counts:
        unique_indices, unique_counts = torch.unique(indices, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes
    else:
        return indices


@torch.no_grad()
def knn_partition(points: torch.Tensor, nodes: torch.Tensor, k: int, return_distance: bool = False):
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
        knn_distances = torch.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
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

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)
    point_limit = min(point_limit, points.shape[0])
    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks



@torch.no_grad()
def point_to_node_partition_bug(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    BUG: this implementation ignores point_to_node indices when building patches. However, the points that do not
    belong to a superpoint should be masked out.


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
    warnings.warn('There is a bug in this implementation. Use `point_to_node_partition` instead.')
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def ball_query_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
):
    node_knn_distances, node_knn_indices = knn_partition(points, nodes, point_limit, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)

    if return_count:
        node_sizes = node_knn_masks.sum(1)  # (N,)
        return node_knn_indices, node_knn_masks, node_sizes
    else:
        return node_knn_indices, node_knn_masks
