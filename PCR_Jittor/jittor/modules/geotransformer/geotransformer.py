import numpy as np
import jittor as jt
import jittor.nn as nn

from PCR_Jittor.jittor.modules.ops import pairwise_distance
from PCR_Jittor.jittor.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer,Mask_RPETransformer2
import time
class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')
    @jt.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = jt.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = jt.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = jt.norm(jt.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = jt.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = jt.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def execute(self, points):
        with jt.no_grad():
            d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings

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
    dists = pairwise_distance(nodes, points)
    knn_distances, knn_indices = dists.topk(dim=1, k=k, largest=False)
    if return_distance:
        return jt.sqrt(knn_distances), knn_indices
    else:
        return knn_indices

def unique_with_inds(x, dim=-1):
    unique, inverse = jt.unique(x, return_inverse=True, dim=dim)
    perm = jt.arange(inverse.size(dim), dtype=inverse.dtype)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

@jt.no_grad()
def cal_geodesic_vectorize(
   query_points, locs_float_, max_step=128, neighbor=64, radius=0.05
):

    locs_float_b = locs_float_
    n_queries=query_points.shape[0]
    n_points = locs_float_b.shape[0]

    #distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

    neighbor = min(neighbor, locs_float_b.shape[0]-1)

    distances_arr,indices_arr = get_knn_indices(locs_float_b,locs_float_b, neighbor,return_distance=True)  # (num_proposal, max_point)
    # NOTE nearest neigbor is themself -> remove first element
    distances_arr = distances_arr[:, 1:]
    indices_arr = indices_arr[:, 1:]

    geo_dist = jt.zeros((n_queries, n_points), dtype=jt.float32) - 1
    visited = jt.zeros((n_queries, n_points), dtype=jt.int32)

    distances, indices = get_knn_indices(locs_float_b,query_points, neighbor,return_distance=True)  # (num_proposal, max_point)

    cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

    sel_indices = jt.nonzero(cond)# (C,) (C,) (C,)
    queries_inds=sel_indices[:,0].reshape(-1)
    neighbors_inds =sel_indices[:,1].reshape(-1)


    #queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
    points_inds = indices[queries_inds, neighbors_inds]  # n_temp
    points_distances = distances[queries_inds, neighbors_inds]  # n_temp

    for step in range(max_step):
        # NOTE find unique indices for each query
        stack_pointquery_inds = jt.stack([points_inds, queries_inds], dim=0)
        _, unique_inds = unique_with_inds(stack_pointquery_inds)

        points_inds = points_inds[unique_inds] 
        queries_inds = queries_inds[unique_inds]
        points_distances = points_distances[unique_inds]

        # NOTE update geodesic and visited look-up table
        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = 1

        # NOTE get new neighbors
        distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
        distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

        # NOTE trick to repeat queries indices for new neighbor
        queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

        # NOTE condition: no visited and radius and indices
        visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
        cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

        # NOTE filter
        #temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2
        sel_indices = jt.nonzero(cond)# (C,) (C,) (C,)
        temp_inds=sel_indices[:,0].reshape(-1)
        neighbors_inds =sel_indices[:,1].reshape(-1)
        if len(temp_inds) == 0:  # no new points:
            break

        points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
        points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
        queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

    return geo_dist


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
    


    def execute(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        """ start_time=time.time()  """ 
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        """ loading_time1 = time.time() - start_time
        start_time=time.time()   """
        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        """ loading_time2 = time.time() - start_time
        print("transformer",loading_time2)
        print("embedding",loading_time1) """
        return ref_feats, src_feats



class CoarseMaskTransformer2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_head, blocks, bin_size_d, bin_size_a, angle_k,max_neighboor,geodesic_radis):
        super(CoarseMaskTransformer2, self).__init__()
        self.bin_size_d = bin_size_d
        self.bin_size_a = bin_size_a
        self.bin_factor_a = 180. / (self.bin_size_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.rde_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rae_proj = nn.Linear(hidden_dim, hidden_dim)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = Mask_RPETransformer2(blocks, hidden_dim, num_head, dropout=0.1, activation_fn='ReLU')
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.rdistance_invarint_proj = nn.Linear(hidden_dim, hidden_dim)
        self.max_neighboors=max_neighboor
        self.bin_size_geo=bin_size_d/2
        self.geodesic_radis=geodesic_radis



    def _get_geometric_geodesic_embeddings(self, points, Radis,geodesic=True):
        with jt.no_grad():
            k = min(self.max_neighboors, points.shape[0])
            dists = pairwise_distance(points, points)
            node_knn_distance, node_knn_indices = dists.topk(dim=1, k=k, largest=False)
            #node_knn_distance,node_knn_indices = get_knn_indices(points, points, self.max_neighboors,return_distance=True)  # (N, max_point)
            node_knn_masks=jt.greater(node_knn_distance, Radis*(0.5))# (N, max_point)
            node_knn_points=points[node_knn_indices]# (N, max_point,3)            
            rde_indices = node_knn_distance / self.bin_size_d
            knn_points =node_knn_points[:,1:4,:]# (N, k, 3)
            ref_vectors = knn_points - points.unsqueeze(1)  # (N, k, 3)
            anc_vectors = node_knn_points.unsqueeze(1) - points.unsqueeze(1).unsqueeze(2)  # (N,1, max_point,3)
            ref_vectors = ref_vectors[node_knn_indices].unsqueeze(1)  # (N,1, max_point, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(anc_vectors.shape[0], anc_vectors.shape[1], anc_vectors.shape[2], self.angle_k, 3)# (N,1, max_point,k,3)
            sin_values = jt.norm(jt.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (N,1, max_point,k)
            cos_values = jt.sum(ref_vectors * anc_vectors, dim=-1)  # (N,1, max_point,k)
            angles = jt.atan2(sin_values, cos_values)  # (N,1, max_point,k)
            rae_indices = angles * self.bin_factor_a

        rde = self.embedding(rde_indices.unsqueeze(1) )  # (N,1, max_point, C)
        rde = self.rde_proj(rde)  # (N,1, max_point, C)

        rae = self.embedding(rae_indices)  # (N,1, max_point,k, C)
        rae = self.rae_proj(rae)  # (N,1, max_point,k, C)
        rae = rae.max(dim=3) # (N,1, max_point, C)

        rge = rde + rae # (N,1, max_point, C)

        if geodesic:
            with jt.no_grad():
                neighboors=min(self.max_neighboors, points.shape[0])
                geo_dists= jt.zeros((len(points), neighboors), dtype=jt.float32)
                geo_dist = cal_geodesic_vectorize(points,points,max_step=32, neighbor=neighboors,radius=self.geodesic_radis)  # (N, num)
                max_geo_dist_context = jt.max(geo_dist, dim=1).reshape(-1)  # (N)
                max_geo_val = jt.max(max_geo_dist_context)
                max_geo_dist_context[max_geo_dist_context < 0] = max_geo_val  # NOTE assign very big value to invalid queries
                max_geo_dist_context = max_geo_dist_context[:, None].expand(geo_dist.shape[0], geo_dist.shape[1])  # (N, num)
                cond = geo_dist < 0
                geo_dist[cond] = max_geo_dist_context[cond]# (N, num)
                for i in range(len(points)):
                    geo_dists[i,:]=geo_dist[i][node_knn_indices[i]] 
                geo_dist=geo_dists
                geo_dists=geo_dists/ self.bin_size_geo
            rdistance_invarint_e = self.embedding(geo_dists.unsqueeze(1) )  # (N,1, max_point, C)
            rdistance_invarint_e = self.rdistance_invarint_proj(rdistance_invarint_e)  # (N,1, max_point, C)
            rdistance_invarint_e+=rde
            return rge,rdistance_invarint_e,node_knn_indices,node_knn_masks,geo_dist#(N,1, max_point, C) (N,1, max_point, C), (N, max_point)， (N, max_point)

        return rge,node_knn_indices,node_knn_masks# (N,1, max_point, C), (N, max_point)， (N, max_point)

    def execute(self, ref_points, src_points, ref_feats, src_feats, Radis, point2trans_indexs=None,ref_masks=None, src_masks=None,gt_corr_indices=None):
        r"""
        Coarse Transformer with Relative Distance Embedding.

        :param ref_points: torch.Tensor (N, 3)
        :param src_points: torch.Tensor (M, 3)
        :param ref_feats: torch.Tensor (N, C)
        :param src_feats: torch.Tensor (M, C)
        :param ref_masks: torch.BoolTensor (N) (default: None)
        :param src_masks: torch.BoolTensor (M) (default: None)
        :return ref_feats: torch.Tensor (N, C)
        :return src_feats: torch.Tensor (M, C)
        """
        
        ref_embeddings,cross_postion_embedding,ref_node_knn_indices ,ref_node_knn_masks,geo_dist= self._get_geometric_geodesic_embeddings(ref_points, Radis,True)
        src_embeddings,src_node_knn_indices ,src_node_knn_masks = self._get_geometric_geodesic_embeddings(src_points, Radis,False)
        # (num_proposal, num_proposal, C),(num_proposal,1, max_point, C), (num_proposal, max_point),(num_proposal, max_point)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        #(N, C),(M, C), (N,1, max_point, C), (M,1, max_point, C), (N, max_point),(M, max_point), (N, max_point),(M, max_point)
        ref_feats, src_feats,pred_masks_list,attn_masks_list,mask_attention_score_list = self.transformer(
            ref_feats, src_feats, ref_embeddings, src_embeddings, cross_postion_embedding,ref_node_knn_indices,src_node_knn_indices,masks0=None, masks1=None
        )
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        return ref_feats, src_feats,ref_node_knn_indices,src_node_knn_indices,geo_dist,pred_masks_list,attn_masks_list,mask_attention_score_list