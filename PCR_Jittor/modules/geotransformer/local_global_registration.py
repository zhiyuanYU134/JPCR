from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import random

from PCR_Jittor.modules.ops import apply_transform,get_knn_indices
from PCR_Jittor.modules.registration import WeightedProcrustes
from PCR_Jittor.modules.ops import index_select

def pairwise_distance(points0, points1, normalized=False, clamp=False):
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

def compute_add_error(gt_transform, est_transform,src_points):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error and Relative Translation Error

    :param gt_transform: torch.Tensor (M,4, 4) 
    :param est_transform: numpy.ndarray (N,4, 4)
    :param src_points: torch.Tensor (n,3) 
    :return rre: float
    :return rte: float
    """
    src_R=pairwise_distance(src_points,src_points).max()
    src_R=torch.sqrt(src_R)
    gt_src_points = apply_transform(src_points.unsqueeze(0), gt_transform)# (M,n,3) 
    est_src_points = apply_transform(src_points.unsqueeze(0), est_transform)# (N,n,3) 
    if gt_transform.is_cuda:
        add_metric=torch.zeros((len(gt_transform),len(est_transform))).cuda()
    else:
        add_metric=torch.zeros((len(gt_transform),len(est_transform)))
    for  i in range(len(gt_transform)):
        gt_src_point=gt_src_points[i]
        gt_src_point=gt_src_point.unsqueeze(0)# (1,n,3) 
        add_metric[i] = torch.sqrt((gt_src_point-est_src_points).abs().pow(2).sum(-1)).mean(-1)# (N) 
    return add_metric/src_R

class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])

        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]


        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            
            batch_transforms= self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform,_,_,_  = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform,_,_,_  = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        torch.set_printoptions(7)
        
        score_mat = torch.exp(score_mat)
        score_mat_ori=score_mat
        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()
        
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        return global_ref_corr_points, global_src_corr_points, global_corr_scores#,score_mat_ori,batch_ref_corr_points, batch_src_corr_points, batch_corr_scores



class FineMatching(nn.Module):
    def __init__(
            self,
            cluster_thre,
            cluster_refine,
            max_num_corr,
            k,
            mutual=False,
            with_slack=False,
            threshold=0.,
            conditional=False,
            matching_radius=0.1,
            min_num_corr=3,
            num_registration_iter=5,
            num_corr_per_patch=16
    ):
        super(FineMatching, self).__init__()
        self.max_num_corr = max_num_corr
        self.k = k
        self.mutual = mutual
        self.with_slack = with_slack
        self.threshold = threshold
        self.conditional = conditional
        self.matching_radius = matching_radius
        self.procrustes = WeightedProcrustes(return_transform=True)
        self.min_num_corr = min_num_corr
        self.num_registration_iter = num_registration_iter
        self.num_corr_per_patch = num_corr_per_patch
        self.cluster_thre=cluster_thre
        self.cluster_refine=cluster_refine

    def compute_score_map_and_corr_map(
            self,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores
    ):
        matching_score_map = torch.exp(matching_score_map)
        corr_mask_map = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        num_proposal, ref_length, src_length = matching_score_map.shape
        proposal_indices = torch.arange(num_proposal).cuda()

        ref_topk_scores, ref_topk_indices = matching_score_map.topk(k=self.k, dim=2)  # (B, N, K)
        ref_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, ref_length, self.k)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(num_proposal, ref_length, self.k)
        ref_score_map = torch.zeros_like(matching_score_map)
        ref_score_map[ref_proposal_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        if self.with_slack:
            ref_score_map = ref_score_map[:, :-1, :-1]
        ref_corr_map = torch.logical_and(torch.gt(ref_score_map, self.threshold), corr_mask_map)

        src_topk_scores, src_topk_indices = matching_score_map.topk(k=self.k, dim=1)  # (B, K, N)
        src_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, self.k, src_length)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(num_proposal, self.k, src_length)
        src_score_map = torch.zeros_like(matching_score_map)
        src_score_map[src_proposal_indices, src_topk_indices, src_indices] = src_topk_scores
        if self.with_slack:
            src_score_map = src_score_map[:, :-1, :-1]
        src_corr_map = torch.logical_and(torch.gt(src_score_map, self.threshold), corr_mask_map)
        score_map = (ref_score_map + src_score_map) / 2

        if self.mutual:
            corr_map = torch.logical_and(ref_corr_map, src_corr_map)
        else:
            corr_map = torch.logical_or(ref_corr_map, src_corr_map)

        if self.conditional:
            node_corr_scores = node_corr_scores.view(-1, 1, 1)
            score_map = score_map * node_corr_scores
        

        return score_map, corr_map
    def cluster_transform(self,estimated_transforms,src_corr_points, ref_corr_points, corr_scores,corr_vote,src_points_m,thresold):
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        estimated_transforms_list=[]
        estimated_transforms_good=[]
        while len(estimated_transforms)>0:
            Rt_score=Rt_scores[0]
            Rt_score_mask=torch.lt(Rt_score,thresold)
            Rt_score_mask_inv=torch.logical_not(Rt_score_mask)
            Rt_scores_tmp=Rt_scores[Rt_score_mask]
            estimated_transforms_tmp=estimated_transforms[Rt_score_mask]
            Rt_scores_tmp=Rt_scores_tmp[:,Rt_score_mask]
            estimated_transforms_good.append(estimated_transforms_tmp[0])
            estimated_transforms_list.append(estimated_transforms_tmp)
            Rt_scores=Rt_scores[Rt_score_mask_inv]
            Rt_scores=Rt_scores[:,Rt_score_mask_inv]
            estimated_transforms=estimated_transforms[Rt_score_mask_inv]
            if len(estimated_transforms)==1:
                estimated_transforms_good.append(estimated_transforms[0])
                estimated_transforms_list.append(estimated_transforms.unsqueeze(0))
                break
        estimated_transforms=torch.full((len(estimated_transforms_good),4,4),0.0).cuda()
        for i in range(len(estimated_transforms)):
            estimated_transforms[i]=estimated_transforms_good[i]
        return estimated_transforms

    def cluster_cal_transform(self,estimated_transforms,src_corr_points, ref_corr_points, corr_scores,corr_vote,src_points_m,thresold):        
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        estimated_transforms_good=[]
        while len(estimated_transforms)>0:
            Rt_score=Rt_scores[0]
            Rt_score_mask=torch.lt(Rt_score,thresold)
            Rt_score_mask_inv=torch.logical_not(Rt_score_mask)
            src_corr_points_tmp=src_corr_points[Rt_score_mask].reshape(-1,3)
            ref_corr_points_tmp=ref_corr_points[Rt_score_mask].reshape(-1,3)
            corr_scores_tmp=corr_scores[Rt_score_mask].reshape(-1)
            estimated_transform =estimated_transforms[0] #self.procrustes(src_corr_points_tmp, ref_corr_points_tmp, corr_scores_tmp)
            for _ in range(2):
                aligned_src_corr_points = apply_transform(src_corr_points_tmp, estimated_transform)
                corr_distances = torch.sum((ref_corr_points_tmp - aligned_src_corr_points) ** 2, dim=1)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                corr_scores_tmp = corr_scores_tmp * inlier_masks.float()
                estimated_transform = self.procrustes(src_corr_points_tmp, ref_corr_points_tmp, corr_scores_tmp)

            estimated_transforms_good.append(estimated_transform)
            Rt_scores=Rt_scores[Rt_score_mask_inv]
            Rt_scores=Rt_scores[:,Rt_score_mask_inv]
            estimated_transforms=estimated_transforms[Rt_score_mask_inv]
            src_corr_points=src_corr_points[Rt_score_mask_inv]
            ref_corr_points=ref_corr_points[Rt_score_mask_inv]
            corr_scores=corr_scores[Rt_score_mask_inv]
            if len(estimated_transforms)==1:
                src_corr_points=src_corr_points.reshape(-1,3)
                ref_corr_points=ref_corr_points.reshape(-1,3)
                corr_scores=corr_scores.reshape(-1)
                estimated_transform =estimated_transforms[0]
                for _ in range(2):
                    aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
                    corr_distances = torch.sum((ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
                    inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                    corr_scores= corr_scores * inlier_masks.float()
                    estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
                estimated_transforms_good.append(estimated_transform)
                break
        if len(estimated_transforms_good)>1:
            estimated_transforms=torch.stack(estimated_transforms_good)
        elif len(estimated_transforms_good)==1:
            estimated_transforms=estimated_transforms_good[0].unsqueeze(0)
        else:
            estimated_transforms=torch.eye(4).unsqueeze(0).cuda()
        return estimated_transforms
    
    def Cal_Inliers(self,estimated_transforms,ref_points_m,src_points_m):
        all_aligned_src_points = apply_transform(src_points_m.unsqueeze(0), estimated_transforms)
        if len(ref_points_m)>self.max_num_corr:
            inds = torch.LongTensor(random.sample(range(len(ref_points_m)), self.max_num_corr)).cuda()
            ref_points_m=ref_points_m[inds]
        inliers=torch.zeros(len(estimated_transforms)).cuda()
        max_instance=8
        head=0
        for i in range((len(estimated_transforms)//max_instance)+1):
            if head+max_instance>len(estimated_transforms):
                end=len(estimated_transforms)
            else:
                end=head+max_instance
            aligned_src_points=all_aligned_src_points[head:end]
            src_closest_distance,src_closest_indices = get_knn_indices(ref_points_m,aligned_src_points.reshape(-1,3),  1,return_distance=True) 
            inlier_masks=torch.lt(src_closest_distance.reshape(-1), self.matching_radius*0.75)
            inlier_masks=inlier_masks.reshape(end-head,len(src_points_m))
            inliers[head:end]=inlier_masks.sum(dim=1)
            head+=max_instance
        return inliers

    def fast_compute_all_transforms(
            self,
            ref_knn_points,
            src_knn_points,
            score_map,
            corr_map,
            ref_points_m,
            src_points_m
    ):
        proposal_indices, ref_indices, src_indices = torch.nonzero(corr_map, as_tuple=True)
        all_ref_corr_points = ref_knn_points[proposal_indices, ref_indices]
        all_src_corr_points = src_knn_points[proposal_indices, src_indices]
        all_corr_scores = score_map[proposal_indices, ref_indices, src_indices]
        #ref_knn_points=torch.unique(ref_knn_points.reshape(-1,3),dim=0)


        """ if all_corr_scores.shape[0] > max_num_corr:
            corr_scores, sel_indices = all_corr_scores.topk(k=max_num_corr, largest=True)
            ref_corr_points = index_select(all_ref_corr_points, sel_indices, dim=0)
            src_corr_points = index_select(all_src_corr_points, sel_indices, dim=0)
        else:
            ref_corr_points = all_ref_corr_points
            src_corr_points = all_src_corr_points
            corr_scores = all_corr_scores """

        ref_corr_points = all_ref_corr_points
        src_corr_points = all_src_corr_points
        corr_scores = all_corr_scores

        # torch.nonzero is row-major, so the correspondences from the same proposal are consecutive.
        # find the first occurrence of each proposal index, then the chunk of this proposal can be obtained.
        corr_node_masks= torch.zeros(corr_map.shape[0],dtype=torch.bool).cuda()
        
        
        unique_masks = torch.ne(proposal_indices[1:], proposal_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [proposal_indices.shape[0]]
        for x, y in zip(unique_indices[:-1], unique_indices[1:]):
            if y - x >= self.min_num_corr:
                corr_node_masks[proposal_indices[y-1]]=True
        chunks = [(x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.min_num_corr]
        num_proposal = len(chunks)
        if num_proposal > 0:
            indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
            stacked_ref_corr_points = index_select(all_ref_corr_points, indices, dim=0)  # (total, 3)
            stacked_src_corr_points = index_select(all_src_corr_points, indices, dim=0)  # (total, 3)
            stacked_corr_scores = index_select(all_corr_scores, indices, dim=0)  # (total,)

            max_corr = np.max([y - x for x, y in chunks])
            target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)

            local_ref_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_ref_corr_points.index_put_([indices0, indices1], stacked_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(num_proposal, max_corr, 3)
            local_src_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_src_corr_points.index_put_([indices0, indices1], stacked_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(num_proposal, max_corr, 3)
            local_corr_scores = torch.zeros(num_proposal * max_corr).cuda()
            local_corr_scores.index_put_([indices], stacked_corr_scores)
            local_corr_scores = local_corr_scores.view(num_proposal, max_corr)


            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)

            aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)

            all_corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(all_corr_distances, (self.matching_radius)  ** 2).float() # (P, N)
            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            inlier_masks=inlier_masks[inlier_index_masks]
            estimated_transforms=estimated_transforms[inlier_index_masks]
            corr_node_masks_ori=corr_node_masks.clone()
            corr_node_masks[corr_node_masks_ori]=inlier_index_masks
          
            cur_corr_scores = local_corr_scores * inlier_masks.float()
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                cur_corr_scores = local_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)

            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            cur_corr_scores=cur_corr_scores[inlier_index_masks]
            
            estimated_transforms=estimated_transforms[inlier_index_masks]
            corr_node_masks_ori=corr_node_masks.clone()
            corr_node_masks[corr_node_masks_ori]=inlier_index_masks

            ref_corr_points = local_ref_corr_points
            src_corr_points = local_src_corr_points
            corr_scores = cur_corr_scores
            
            num_proposal, max_corr, _=local_ref_corr_points.shape
            local_ref_corr_points = local_ref_corr_points.reshape(num_proposal*max_corr,3)
            local_src_corr_points = local_src_corr_points.reshape(num_proposal*max_corr,3)
            local_corr_scores = local_corr_scores.reshape(num_proposal*max_corr)
            local_corr_masks=torch.gt(local_corr_scores,0)
            local_src_corr_points=local_src_corr_points[local_corr_masks]
            local_ref_corr_points=local_ref_corr_points[local_corr_masks]
            local_corr_scores=local_corr_scores[local_corr_masks]
            aligned_src_corr_points = apply_transform(local_src_corr_points.unsqueeze(0), estimated_transforms)
            corr_distances = torch.sum((local_ref_corr_points.unsqueeze(0) - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(corr_distances, (self.matching_radius/2) ** 2).float()
            corr_vote=inlier_masks.sum(dim=1)
            #corr_vote=(local_corr_scores.unsqueeze(0) * inlier_masks).sum(dim=1)
            estimated_transforms_ori=estimated_transforms
            sorted_inlier,inlier_indices=torch.sort(corr_vote,descending=True)
            estimated_transforms=estimated_transforms[inlier_indices]
            corr_vote=corr_vote[inlier_indices]
            ref_corr_points = ref_corr_points[inlier_indices]
            src_corr_points = src_corr_points[inlier_indices]
            corr_scores = corr_scores[inlier_indices]
            
            estimated_transforms=self.cluster_cal_transform(estimated_transforms,src_corr_points, ref_corr_points, corr_scores,corr_vote,src_points_m,self.cluster_thre)
            if self.cluster_refine:
                inliers=self.Cal_Inliers(estimated_transforms,ref_points_m,src_points_m)
                max_inliers=torch.max(inliers)
                estimated_transforms_score_mask=inliers>max_inliers*0.8
                estimated_transforms=estimated_transforms[estimated_transforms_score_mask]
            return estimated_transforms,estimated_transforms_ori,corr_node_masks,local_ref_corr_points,local_src_corr_points,local_corr_scores
        else:
            estimated_transforms = self.procrustes(src_corr_points, ref_corr_points, corr_scores)

            aligned_src_corr_points = apply_transform(src_corr_points, estimated_transforms)
            corr_distances = torch.sum((ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
            inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2).float()
            cur_corr_scores = corr_scores * inlier_masks
            estimated_transforms = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(src_corr_points, estimated_transforms)
                corr_distances = torch.sum((ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
                inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2)
                cur_corr_scores = corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
            inlier_ratio=torch.sum(inlier_masks)
            return estimated_transforms.unsqueeze(0),estimated_transforms.unsqueeze(0),corr_node_masks,all_ref_corr_points,all_src_corr_points,all_corr_scores

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores,
            ref_points_m,
            src_points_m
    ):
        """
        :param ref_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param src_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param ref_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param src_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param matching_score_map: torch.Tensor (num_proposal, num_point, num_point)
        :param node_corr_scores: torch.Tensor (num_proposal)

        :return ref_corr_indices: torch.LongTensor (self.num_corr,)
        :return src_corr_indices: torch.LongTensor (self.num_corr,)
        :return corr_scores: torch.Tensor (self.num_corr,)
        """
        score_map, corr_map = self.compute_score_map_and_corr_map(
            ref_knn_masks, src_knn_masks, matching_score_map, node_corr_scores
        )

        estimated_transforms,estimated_transforms_ori,corr_node_masks,all_ref_corr_points,all_src_corr_points,all_corr_scores= self.fast_compute_all_transforms(
                ref_knn_points, src_knn_points, score_map, corr_map,ref_points_m,src_points_m
            )
        return estimated_transforms,estimated_transforms_ori,corr_node_masks,all_ref_corr_points,all_src_corr_points,all_corr_scores#score_map, corr_map,


class InstanceAwarePointMatching(nn.Module):
    def __init__(
            self,
            cluster_thre,
            cluster_refine,
            max_num_corr,
            k,
            mutual=False,
            with_slack=False,
            threshold=0.,
            conditional=False,
            matching_radius=0.1,
            min_num_corr=3,
            num_registration_iter=5
    ):
        super(InstanceAwarePointMatching, self).__init__()
        self.max_num_corr = max_num_corr
        self.k = k
        self.mutual = mutual
        self.with_slack = with_slack
        self.threshold = threshold
        self.conditional = conditional
        self.matching_radius = matching_radius
        self.procrustes = WeightedProcrustes(return_transform=True)
        self.min_num_corr = min_num_corr
        self.num_registration_iter = num_registration_iter
        self.cluster_thre=cluster_thre
        self.cluster_refine=cluster_refine

    def compute_score_map_and_corr_map(
            self,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores
    ):
        matching_score_map = torch.exp(matching_score_map)
        corr_mask_map = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        num_proposal, ref_length, src_length = matching_score_map.shape
        proposal_indices = torch.arange(num_proposal).cuda()

        ref_topk_scores, ref_topk_indices = matching_score_map.topk(k=self.k, dim=2)  # (B, N, K)
        ref_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, ref_length, self.k)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(num_proposal, ref_length, self.k)
        ref_score_map = torch.zeros_like(matching_score_map)
        ref_score_map[ref_proposal_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        if self.with_slack:
            ref_score_map = ref_score_map[:, :-1, :-1]
        ref_corr_map = torch.logical_and(torch.gt(ref_score_map, self.threshold), corr_mask_map)

        src_topk_scores, src_topk_indices = matching_score_map.topk(k=self.k, dim=1)  # (B, K, N)
        src_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, self.k, src_length)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(num_proposal, self.k, src_length)
        src_score_map = torch.zeros_like(matching_score_map)
        src_score_map[src_proposal_indices, src_topk_indices, src_indices] = src_topk_scores
        if self.with_slack:
            src_score_map = src_score_map[:, :-1, :-1]
        src_corr_map = torch.logical_and(torch.gt(src_score_map, self.threshold), corr_mask_map)
        score_map = (ref_score_map + src_score_map) / 2

        if self.mutual:
            corr_map = torch.logical_and(ref_corr_map, src_corr_map)
        else:
            corr_map = torch.logical_or(ref_corr_map, src_corr_map)

        if self.conditional:
            node_corr_scores = node_corr_scores.view(-1, 1, 1)
            score_map = score_map * node_corr_scores
        
        return score_map, corr_map

    def cluster_transform(self,estimated_transforms,src_corr_points, ref_corr_points, corr_scores,corr_vote,src_points_m,thresold):
        #select candidates which are isolated        
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        Rt_score_mask=torch.lt(Rt_scores,thresold)
        index=torch.nonzero(Rt_score_mask).reshape(-1)
        index=torch.unique(index)
        est_trans_mask=torch.zeros(len(estimated_transforms))
        est_trans_mask[index]=1
        est_trans_mask=est_trans_mask==1
        est_trans_mask_inv=torch.logical_not(est_trans_mask)
        estimated_transforms_inv=estimated_transforms[est_trans_mask_inv]

        
        #candidate selection  
        estimated_transforms=estimated_transforms[est_trans_mask]
        src_corr_points=src_corr_points[est_trans_mask]
        ref_corr_points=ref_corr_points[est_trans_mask]
        corr_scores=corr_scores[est_trans_mask]
        corr_vote=corr_vote[est_trans_mask]
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        all_ref_corr_points = torch.zeros((1,3)).cuda()
        all_src_corr_points = torch.zeros((1,3)).cuda()
        all_corr_scores = torch.zeros((1)).cuda()

        estimated_transforms_good=[]
        len_corr=[]
        while len(estimated_transforms)>0:
            Rt_score=Rt_scores[0]
            Rt_score_mask=torch.lt(Rt_score,thresold)
            Rt_score_mask_inv=torch.logical_not(Rt_score_mask)
            src_corr_points_tmp=src_corr_points[Rt_score_mask].reshape(-1,3)
            ref_corr_points_tmp=ref_corr_points[Rt_score_mask].reshape(-1,3)
            len_corr.append(len(ref_corr_points_tmp))
            corr_scores_tmp=corr_scores[Rt_score_mask].reshape(-1)
            all_ref_corr_points=torch.cat((all_ref_corr_points,ref_corr_points_tmp),dim=0)
            all_src_corr_points=torch.cat((all_src_corr_points,src_corr_points_tmp),dim=0)
            all_corr_scores=torch.cat((all_corr_scores,corr_scores_tmp),dim=0)
            estimated_transform =estimated_transforms[0] 
            estimated_transforms_good.append(estimated_transform)
            Rt_scores=Rt_scores[Rt_score_mask_inv]
            Rt_scores=Rt_scores[:,Rt_score_mask_inv]
            estimated_transforms=estimated_transforms[Rt_score_mask_inv]
            src_corr_points=src_corr_points[Rt_score_mask_inv]
            ref_corr_points=ref_corr_points[Rt_score_mask_inv]
            corr_scores=corr_scores[Rt_score_mask_inv]
            if len(estimated_transforms)==1:
                
                src_corr_points=src_corr_points.reshape(-1,3)
                ref_corr_points=ref_corr_points.reshape(-1,3)
                corr_scores=corr_scores.reshape(-1)
                len_corr.append(len(ref_corr_points))
                all_ref_corr_points=torch.cat((all_ref_corr_points,ref_corr_points),dim=0)
                all_src_corr_points=torch.cat((all_src_corr_points,src_corr_points),dim=0)
                all_corr_scores=torch.cat((all_corr_scores,corr_scores),dim=0)
                estimated_transform =estimated_transforms[0]
                estimated_transforms_good.append(estimated_transform)
                break
        #refine candidate
        if len(estimated_transforms_good)>1:
            estimated_transforms=torch.stack(estimated_transforms_good)
            max_corr=np.max(np.array(len_corr))
            all_ref_corr_points=all_ref_corr_points[1:,:]
            all_src_corr_points=all_src_corr_points[1:,:]
            all_corr_scores=all_corr_scores[1:]
            local_ref_corr_points = torch.zeros(len(len_corr)*max_corr, 3).cuda()
            local_src_corr_points = torch.zeros(len(len_corr)*max_corr, 3).cuda()
            local_corr_scores = torch.zeros(len(len_corr)*max_corr).cuda()
            target_chunks = [(i * max_corr, i * max_corr + y ) for i, (y) in enumerate(len_corr)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)
            local_ref_corr_points.index_put_([indices0, indices1], all_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(len(len_corr), max_corr, 3)
            local_src_corr_points.index_put_([indices0, indices1], all_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(len(len_corr), max_corr, 3)
            local_corr_scores.index_put_([indices], all_corr_scores)
            local_corr_scores = local_corr_scores.view(len(len_corr), max_corr)
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            for _ in range(4):
                aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                local_corr_scores = local_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            estimated_transforms=torch.cat((estimated_transforms,estimated_transforms_inv),dim=0)
        elif len(estimated_transforms_good)==1:
            estimated_transforms=estimated_transforms_good[0].unsqueeze(0)
            estimated_transforms=torch.cat((estimated_transforms,estimated_transforms_inv),dim=0)
        else:
            estimated_transforms=estimated_transforms_inv#torch.eye(4).unsqueeze(0).cuda()
        
        return estimated_transforms
    
    def Cal_Inliers(self,estimated_transforms,ref_points_m,src_points_m):
        all_aligned_src_points = apply_transform(src_points_m.unsqueeze(0), estimated_transforms)
        if len(ref_points_m)>self.max_num_corr:
            inds = torch.LongTensor(random.sample(range(len(ref_points_m)), self.max_num_corr)).cuda()
            ref_points_m=ref_points_m[inds]
        inliers=torch.zeros(len(estimated_transforms)).cuda()
        max_instance=8
        head=0
        for i in range((len(estimated_transforms)//max_instance)+1):
            if head+max_instance>len(estimated_transforms):
                end=len(estimated_transforms)
            else:
                end=head+max_instance
            aligned_src_points=all_aligned_src_points[head:end]
            src_closest_distance,src_closest_indices = get_knn_indices(ref_points_m,aligned_src_points.reshape(-1,3),  1,return_distance=True) 
            inlier_masks=torch.lt(src_closest_distance.reshape(-1), self.matching_radius)
            inlier_masks=inlier_masks.reshape(end-head,len(src_points_m))
            inliers[head:end]=inlier_masks.sum(dim=1)
            head+=max_instance
        return inliers

    def fast_compute_all_transforms(
            self,
            ref_knn_points,
            src_knn_points,
            score_map,
            corr_map,
            ref_points_m,
            src_points_m
    ):
        proposal_indices, ref_indices, src_indices = torch.nonzero(corr_map, as_tuple=True)
        all_ref_corr_points = ref_knn_points[proposal_indices, ref_indices]
        all_src_corr_points = src_knn_points[proposal_indices, src_indices]
        all_corr_scores = score_map[proposal_indices, ref_indices, src_indices]

        # torch.nonzero is row-major, so the correspondences from the same proposal are consecutive.
        # find the first occurrence of each proposal index, then the chunk of this proposal can be obtained.
        unique_masks = torch.ne(proposal_indices[1:], proposal_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [proposal_indices.shape[0]]

        chunks = [(x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.min_num_corr]
        num_proposal = len(chunks)
        if num_proposal > 0:
            indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
            stacked_ref_corr_points = index_select(all_ref_corr_points, indices, dim=0)  # (total, 3)
            stacked_src_corr_points = index_select(all_src_corr_points, indices, dim=0)  # (total, 3)
            stacked_corr_scores = index_select(all_corr_scores, indices, dim=0)  # (total,)

            max_corr = np.max([y - x for x, y in chunks])
            target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)

            local_ref_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_ref_corr_points.index_put_([indices0, indices1], stacked_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(num_proposal, max_corr, 3)
            local_src_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_src_corr_points.index_put_([indices0, indices1], stacked_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(num_proposal, max_corr, 3)
            local_corr_scores = torch.zeros(num_proposal * max_corr).cuda()
            local_corr_scores.index_put_([indices], stacked_corr_scores)
            local_corr_scores = local_corr_scores.view(num_proposal, max_corr)

            #calculate candidate's pose
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
            corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(corr_distances, (self.matching_radius)  ** 2).float() # (P, N)
            
            #remove bad poses
            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            inlier_masks=inlier_masks[inlier_index_masks]
            estimated_transforms=estimated_transforms[inlier_index_masks]
            
            #refine candidate's pose
            cur_corr_scores = local_corr_scores * inlier_masks.float()
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                cur_corr_scores = local_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)

            #remove bad poses
            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            cur_corr_scores=cur_corr_scores[inlier_index_masks]
            estimated_transforms=estimated_transforms[inlier_index_masks]

            #remove bad correspondences
            num_proposal, max_corr, _=local_ref_corr_points.shape
            all_ref_corr_points = local_ref_corr_points.reshape(num_proposal*max_corr,3)
            all_src_corr_points = local_src_corr_points.reshape(num_proposal*max_corr,3)
            all_corr_scores = local_corr_scores.reshape(num_proposal*max_corr)
            all_corr_masks=torch.gt(all_corr_scores,0)
            all_ref_corr_points=all_ref_corr_points[all_corr_masks]
            all_src_corr_points=all_src_corr_points[all_corr_masks]
            all_corr_scores=all_corr_scores[all_corr_masks]

            #sort poses by inliers
            aligned_src_corr_points = apply_transform(all_src_corr_points.unsqueeze(0), estimated_transforms)
            corr_distances = torch.sum((all_ref_corr_points.unsqueeze(0) - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2).float()
            corr_vote=inlier_masks.sum(dim=1)
            sorted_inlier,inlier_indices=torch.sort(corr_vote,descending=True)
            estimated_transforms=estimated_transforms[inlier_indices]
            corr_vote=corr_vote[inlier_indices]
            local_ref_corr_points = local_ref_corr_points[inlier_indices]
            local_src_corr_points = local_src_corr_points[inlier_indices]
            cur_corr_scores = cur_corr_scores[inlier_indices]

            estimated_transforms=self.cluster_transform(estimated_transforms,local_src_corr_points, local_ref_corr_points, cur_corr_scores,corr_vote,src_points_m,self.cluster_thre)
            #remove bad poses
            if self.cluster_refine:
                if len(estimated_transforms)>0:
                    inliers=self.Cal_Inliers(estimated_transforms,ref_points_m,src_points_m)
                    max_inliers=torch.max(inliers)
                    estimated_transforms_score_mask=inliers>max_inliers*0.8
                    estimated_transforms=estimated_transforms[estimated_transforms_score_mask]
            return estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores
        else:
            estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, all_corr_scores)

            aligned_src_corr_points = apply_transform(all_src_corr_points, estimated_transforms)
            corr_distances = torch.sum((all_ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
            inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2).float()
            cur_corr_scores = all_corr_scores * inlier_masks
            estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(all_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((all_ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
                inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2)
                cur_corr_scores = all_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, cur_corr_scores)
            return estimated_transforms.unsqueeze(0),all_ref_corr_points,all_src_corr_points,all_corr_scores

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores,
            ref_points_m,
            src_points_m
            
    ):
        """
        :param ref_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param src_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param ref_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param src_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param matching_score_map: torch.Tensor (num_proposal, num_point, num_point)
        :param node_corr_scores: torch.Tensor (num_proposal)

        :return estimated_transforms: torch.LongTensor (N,4,4)
        :return ref_corr_indices: torch.LongTensor (self.num_corr,)
        :return src_corr_indices: torch.LongTensor (self.num_corr,)
        :return corr_scores: torch.Tensor (self.num_corr,)
        """
        score_map, corr_map = self.compute_score_map_and_corr_map(
            ref_knn_masks, src_knn_masks, matching_score_map, node_corr_scores
        )
        estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores= self.fast_compute_all_transforms(
                ref_knn_points, src_knn_points, score_map, corr_map,ref_points_m,src_points_m
            )
        return estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores