""" import torch
from torch import Tensor
from typing import Optional, Tuple, Union
from pykeops.torch import LazyTensor """
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
from typing import Optional, Tuple, Union

def apply_transform(points, transform, normals= None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = jt.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points=points.expand(rotation.shape[0],points.shape[1], points.shape[2]) 

        points = jt.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
    elif transform.dim() == 3 and points.dim() == 2:
        # case 3: (B, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points.unsqueeze(1)
        points = jt.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.squeeze(1)
        if normals is not None:
            normals = normals.unsqueeze(1)
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.squeeze(1)
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points
def apply_deformation(
    points,
    nodes,
    transforms,
    anchor_indices,
    anchor_weights,
    eps = 1e-6,
):
    """Apply Embedded Deformation Warping Function.

    Args:
        points (Tensor): The points to warp in the shape of (N, 3).
        nodes (Tensor): The graph nodes in the shape of (M, 3).
        transforms (Tensor): The associated transformations for each node in the shape of (M, 4, 4).
        anchor_indices (LongTensor): The indices of the anchor nodes for the points in the shape of (N, K). If an index
            is -1, the corresponding anchor does not exist.
        anchor_weights (Tensor): The skinning weights of the anchor nodes for the point in the shape of (N, K).
        eps (float=1e-6): A safe number for division.

    Returns:
        warped_points (Tensor): The warped points in the shape of (N, 3).
    """
    anchor_weights = anchor_weights / (anchor_weights.sum(dim=1, keepdim=True) + eps)  # (N, K)
    anchor_masks = jt.not_equal(anchor_indices, -1)  # (N, K)
    sel_indices = jt.nonzero(
            anchor_masks
        )
    p_indices=sel_indices[:,0].reshape(-1)
    col_indices =sel_indices[:,1].reshape(-1)
    n_indices = anchor_indices[p_indices, col_indices]  # (C)
    weights = anchor_weights[p_indices, col_indices]  # (C)
    sel_points = points[p_indices]  # (C, 3)
    sel_nodes = nodes[n_indices]  # (C, 3)
    sel_transforms = transforms[n_indices]  # (C, 4, 4)
    sel_warped_points = apply_transform(sel_points - sel_nodes, sel_transforms) + sel_nodes  # (C, 3)
    sel_warped_points = sel_warped_points * weights.unsqueeze(1)  # (C, 3)
    warped_points = jt.zeros_like(points)  # (N, 3)
    p_indices = p_indices#.unsqueeze(1).expand(sel_warped_points.shape)  # (C, 3)
    for j in range(len(p_indices)):
        warped_points[p_indices[j]]+=sel_warped_points[j]
    #jt.index_add_(warped_points,0, p_indices, sel_warped_points)  # (N, 3)#danger
    return warped_points
def compute_skinning_weights(distances, node_coverage):
    """Skinning weight proposed in DynamicFusion.

    w = exp(-d^2 / (2 * r^2))

    Args:
        distances (Tensor): The distance tensor in arbitrary shape.
        node_coverage (float): The node coverage.

    Returns:
        weights (Tensor): The skinning weights in arbitrary shape.
    """
    weights = jt.exp(-(distances ** 2) / (2.0 * node_coverage ** 2))
    return weights


def build_euclidean_deformation_graph(
    points,
    nodes,
    num_anchors: int,
    node_coverage: float,
    return_point_anchor: bool = True,
    return_node_graph: bool = True,
    return_distance: bool = False,
    return_adjacent_matrix: bool = False,
    eps: float = 1e-6,
):
    """Build deformation graph with euclidean distance.

    We use the method proposed in Embedded Deformation to construct the deformation graph:
        1. Each point is assigned to its k-nearest nodes.
        2. If two nodes cover the same point, there is an edge between them.
        3. We compute the skinning weights following DynamicFusion.

    Args:
        points (Tensor): the points in the shape of (N, 3).
        nodes (Tensor): the nodes in the shape of (M, 3).
        num_anchors (int): the number of anchors for each point.
        node_coverage (float): the node coverage to compute skinning weights.
        return_point_anchor (bool): if True, return the anchors for the points. Default: True.
        return_node_graph (bool): if True, return the edges between the nodes. Default: True.
        return_distance (bool): if True, return the distance. Default: False.
        return_adjacent_matrix (bool): if True, return the adjacent matrix instead of edge list. Default: False.
            Only take effect when 'return_node_graph' is True.
        eps (float): A safe number for division. Default: 1e-6.

    Returns:
        A LongTensor of the anchor node indices for the points in the shape of (N, K).
        A Tensor of the anchor node weights for the points in the shape of (N, K).
        A Tensor of the anchor node distances for the points in the shape of (N, K).
        A LongTensor of the endpoint indices of the edges in the shape of (E, 2).
        A Tensor of the weights of the edges in the shape of (E).
        A Tensor of the distances of the edges in the shape of (E).
        A BoolTensor of the adjacent matrix between nodes in the shape of (M, M).
        A Tensor of the skinning weight matrix between nodes of (M, M).
        A Tensor of the distance matrix between nodes of (M, M).
    """
    output_list = []

    anchor_distances, anchor_indices = knn(points, nodes, num_anchors, return_distance=True)  # (N, K)
    anchor_weights = compute_skinning_weights(anchor_distances, node_coverage)  # (N, K)
    anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)  # (N, K)

    if return_point_anchor:
        output_list.append(anchor_indices)
        output_list.append(anchor_weights)
        if return_distance:
            output_list.append(anchor_distances)

    if return_node_graph:
        point_indices = jt.arange(points.shape[0]).unsqueeze(1).expand(anchor_indices.shape)  # (N, K)
        node_to_point = jt.zeros((nodes.shape[0], points.shape[0]))  # (N, M)
        node_to_point[anchor_indices, point_indices] = 1.0
        adjacent_mat = jt.greater(jt.linalg.einsum("nk,mk->nm", node_to_point, node_to_point), 0)
        distance_mat = pairwise_distance(nodes, nodes, squared=False)
        weight_mat = compute_skinning_weights(distance_mat, node_coverage)
        weight_mat = weight_mat * adjacent_mat.float()
        weight_mat = weight_mat / weight_mat.sum(dim=-1, keepdim=True).clamp(min_v=eps)
        if return_adjacent_matrix:
            output_list.append(adjacent_mat)
            output_list.append(weight_mat)
            if return_distance:
                distance_mat = distance_mat * adjacent_mat.float()
                output_list.append(distance_mat)
        else:
            edge_indices = jt.nonzero(adjacent_mat).contiguous()
            edge_weights = weight_mat[adjacent_mat].contiguous()
            output_list.append(edge_indices)
            output_list.append(edge_weights)
            if return_distance:
                edge_distances = distance_mat[adjacent_mat].contiguous()
                output_list.append(edge_distances)

    return tuple(output_list)
def index_select(data, index, dim):
    output=jt.index_select(data,dim,index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)
    return output


def pairwise_distance(
    x,
    y,
    normalized = False,
    transposed = False,
    squared = True,
    strict= False,
    eps = 1e-8,
):
    """Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): the row tensor in the shape of (*, N, C).
        y (Tensor): the column tensor in the shape of (*, M, C).
        normalized (bool): if the points are normalized, we have "x^2 + y^2 = 1", so "d^2 = 2 - 2xy". Default: False.
        transposed (bool): if True, x and y are in the shapes of (*, C, N) and (*, C, M) respectively. Default: False.
        squared (bool): if True, return squared distance. Default: True.
        strict (bool): if True, use strict mode to guarantee precision. Default: False.
        eps (float): a safe number for sqrt. Default: 1e-8.

    Returns:
        dist: Tensor (*, N, M)
    """
    if strict:
        if transposed:
            displacements = x.unsqueeze(-1) - y.unsqueeze(-2)  # (*, C, N, 1) x (*, C, 1, M) -> (*, C, N, M)
            distances = jt.norm(displacements, dim=-3)  # (*, C, N, M) -> (*, N, M)
        else:
            displacements = x.unsqueeze(-2) - y.unsqueeze(-3)  # (*, N, 1, C) x (*, 1, M, C) -> (*, N, M, C)
            distances = jt.norm(displacements, dim=-1)  # (*, N, M, C) -> (*, N, M)

        if squared:
            distances = distances.pow(2)
    else:
        if transposed:
            channel_dim = -2
            xy = jt.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
        else:
            channel_dim = -1
            xy = jt.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]

        if normalized:
            distances = 2.0 - 2.0 * xy
        else:
            x2 = jt.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
            y2 = jt.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
            distances = x2 - 2 * xy + y2

        distances = distances.clamp(min_v=0.0)

        if not squared:
            distances = jt.sqrt(distances + eps)

    return distances


def compute_nonrigid_feature_matching_recall(
    src_corr_points,
    tgt_corr_points,
    src_points,
    scene_flows,
    test_indices,
    transform= None,
    acceptance_radius = 0.04,
    distance_limit = 0.1,
):
    """Non-rigid Feature Matching Recall for 4DMatch.

    Args:
        src_corr_points (Tensor): (N, 3)
        tgt_corr_points (Tensor): (N, 3)
        src_points (Tensor): (M, 3)
        scene_flows (Tensor): (M, 3)
        test_indices (LongTensor): (K)
        transform (Tensor, optional): (4, 4)
        acceptance_radius (float=0.04): acceptance radius
        distance_limit (float=0.1): max distance for scene flow interpolation

    Returns:
        recall (Tensor): non-rigid feature matching recall
    """
    corr_motions = tgt_corr_points - src_corr_points  # (N, 3)
    src_test_points = src_points[test_indices]  # (K, 3)
    pred_motions = knn_interpolate(src_test_points, src_corr_points, corr_motions, k=3, distance_limit=distance_limit)
    pred_tgt_test_points = src_test_points + pred_motions  # (K, 3)
    gt_scene_flows = scene_flows[test_indices]  # (K, 3)
    gt_tgt_test_points = apply_transform(src_test_points + gt_scene_flows, transform)
    residuals = jt.norm(pred_tgt_test_points - gt_tgt_test_points, dim=1)
    recall = jt.less(residuals, acceptance_radius).float().mean()#danger
    recall[jt.isnan(recall)]=0
    return recall


def compute_scene_flow_accuracy(
    inputs,
    targets,
    acceptance_absolute_error: float,
    acceptance_relative_error: float,
    eps = 1e-20,
):
    absolute_errors =jt.norm(inputs - targets, dim=1)
    target_lengths =jt.norm(targets, dim=1)
    relative_errors = absolute_errors / (target_lengths + eps)
    absolute_results = jt.less(absolute_errors, acceptance_absolute_error)
    relative_results = jt.less(relative_errors, acceptance_relative_error)
    results = jt.logical_or(absolute_results, relative_results)
    accuracy = results.float().mean()
    return accuracy


def compute_scene_flow_outlier_ratio(
    inputs,
    targets,
    acceptance_absolute_error: Optional[float],
    acceptance_relative_error: Optional[float],
    eps: float = 1e-20,
):
    absolute_errors = jt.norm(inputs - targets, dim=1)
    target_lengths = jt.norm(targets, dim=1)
    relative_errors = absolute_errors / (target_lengths + eps)
    results = inputs.new_zeros(size=(inputs.shape[0],)).bool()
    if acceptance_absolute_error is not None:
        results = jt.logical_or(results, jt.greater(absolute_errors, acceptance_absolute_error))
    if acceptance_relative_error is not None:
        results = jt.logical_or(results, jt.greater(relative_errors, acceptance_relative_error))
    outlier_ratio = results.float().mean()
    return outlier_ratio



def evaluate_binary_classification(
    inputs, targets, positive_threshold = 0.5, use_logits= False, eps= 1e-6
):
    """Binary classification precision and recall metric.

    Args:
        inputs (Tensor): inputs (*)
        targets (Tensor): targets, 0 or 1 (*)
        positive_threshold (float=0.5): considered as positive if larger than this value.
        use_logits (bool=False): If True, the inputs are logits, a sigmoid is needed.
        eps (float=1e-6): safe number.

    Return:
        precision (Tensor): precision
        recall (Tensor): recall
    """
    if use_logits:
        inputs = jt.sigmoid(inputs)

    targets = targets.float()

    results = jt.greater(inputs, positive_threshold).float()
    correct_results = results * targets

    precision = correct_results.sum() / (results.sum() + eps)
    recall = correct_results.sum() / (targets.sum() + eps)

    return precision, recall

def knn_interpolate(
    q_points,
    s_points,
    s_feats,
    k = 3,
    eps = 1e-10,
    distance_limit: Optional[float] = None,
):
    """kNN interpolate.

    Args:
        q_points (Tensor): a Tensor of the query points in the shape of (M, 3).
        s_points (Tensor): a Tensor of the support points in the shape of (N, 3).
        s_feats (Tensor): a Tensor of the support features in the shape of (N, C).
        k (int): the number of the neighbors. Default: 3.
        eps (float): the safe number for division. Default: 1e-10.
        distance_limit (float, optional): the distance limit for the neighbors. If not None, the neighbors further than
            this distance are ignored.

    Returns:
        A Tensor of the features of the query points in the shape of (M, C).
    """
    knn_distances, knn_indices = keops_knn(q_points, s_points, k)  # (M, K)
    if distance_limit is not None:
        masks =jt.greater(knn_distances, distance_limit)  # (M, K)
        knn_distances[masks]=1e10
    weights = 1.0 / (knn_distances + eps)  # (M, K)
    weights = weights / weights.sum(dim=1, keepdims=True)  # (M, K)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (M, K, C)
    q_feats = (knn_feats * weights.unsqueeze(-1)).sum(dim=1)  # (M, C)
    return q_feats

def keops_knn(q_points, s_points, k: int):
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2

    dists = pairwise_distance(q_points, s_points)
    
    knn_distances, knn_indices = dists.topk(dim=num_batch_dims + 1, k=k, largest=False)
    
    """ xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K) """
    return jt.sqrt(knn_distances), knn_indices


def knn(
    q_points,
    s_points,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Use KeOps to accelerate computation.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are ignored according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): the padding mode for neighbors further than distance radius. ('nearest', 'empty').
        padding_value (float=1e10): the value for padding.
        squeeze (bool=False): if True, the distance and the indices are squeezed if k=1.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = jt.greater_equal(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = jt.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = jt.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points

    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices

def spatial_consistency(src_corr_points, tgt_corr_points, sigma: float):
    """Compute spatial consistency.

    SC_{i,j} = max(1 - d_{i,j}^2 / sigma ^2, 0)
    d_{i,j} = \lvert \lVert p_i - p_j \rVert - \lVert q_i - q_j \rVert \rvert

    Args:
        src_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        tgt_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        sigma (float): The sensitivity factor.

    Returns:
        A Tensor of the spatial consistency between the correspondences in the shape of (*, N, N).
    """
    src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = jt.abs(src_dist_mat - tgt_dist_mat)
    consistency_mat = nn.relu(1.0 - delta_mat.pow(2) / (sigma ** 2))
    return consistency_mat
