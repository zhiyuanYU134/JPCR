import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
from PCR_Jittor.jittor.modules.ops import pairwise_distance
import time
class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def execute(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
         
        if ref_masks is None:
            ref_masks = jt.ones(size=(ref_feats.shape[0],), dtype='bool')
        if src_masks is None:
            src_masks = jt.ones(size=(src_feats.shape[0],), dtype='bool')
        # remove empty patch
        """ print(ref_masks.reshape(-1))
        print(jt.nonzero(ref_masks).shape) """
        """ start_time=time.time() """
        ref_indices = jt.nonzero(ref_masks).reshape(-1)
        src_indices = jt.nonzero(src_masks).reshape(-1)
        """ savename ='ref_masks.npz'
        np.savez(savename, 
        ref_masks=ref_masks.cpu().numpy()) 
        savename ='src_masks.npz'
        np.savez(savename, 
        src_masks=src_masks.cpu().numpy()) 
        savename ='ref_feats.npz'
        np.savez(savename, 
        ref_feats=ref_feats.cpu().numpy()) 
        savename ='src_feats.npz'
        np.savez(savename, 
        src_feats=src_feats.cpu().numpy())  """
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        """ loading_time = time.time() - start_time
        print("nonzero",loading_time)
        start_time=time.time()  """
        # select top-k proposals

        matching_scores = jt.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdims=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdims=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        """ loading_time = time.time() - start_time
        print("topk",loading_time) """
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores

class CoarseMatching(nn.Module):
    def __init__(
            self,
            num_proposal,
            dual_softmax=True
    ):
        super(CoarseMatching, self).__init__()
        self.num_proposal = num_proposal
        self.dual_softmax = dual_softmax

    def execute(self, ref_feats, src_feats):
        # remove empty node
        """ ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = index_select(ref_feats, ref_indices, dim=0)
        src_feats = index_select(src_feats, src_indices, dim=0) """
        # select top-k proposals
        matching_scores = jt.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_softmax:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdims=True)
            #src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * matching_scores
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=self.num_proposal, largest=True)
        ref_corr_indices = corr_indices // matching_scores.shape[1]
        src_corr_indices = corr_indices % matching_scores.shape[1]
        #all_ref_corr_indices,_=jt.argmax(matching_scores,dim=1)

        
        return ref_corr_indices, src_corr_indices, corr_scores#,all_ref_corr_indices