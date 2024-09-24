import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat


class SuperPointTargetGenerator(nn.Module):
    def __init__(self, num_targets, overlap_threshold):
        super(SuperPointTargetGenerator, self).__init__()
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold

    @jt.no_grad()
    def execute(self, gt_corr_indices, gt_corr_overlaps):
        r"""Generate ground truth superpoint (patch) correspondences.

        Randomly select "num_targets" correspondences whose overlap is above "overlap_threshold".

        Args:
            gt_corr_indices (LongTensor): ground truth superpoint correspondences (N, 2)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)

        Returns:
            gt_ref_corr_indices (LongTensor): selected superpoints in reference point cloud.
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        gt_corr_masks = jt.greater(gt_corr_overlaps, self.overlap_threshold)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]

        if gt_corr_indices.shape[0] > self.num_targets:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_targets, replace=False)
            sel_indices = jt.Var(sel_indices)
            gt_corr_indices = gt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps



class CoarseTargetGenerator(nn.Module):
    def __init__(self, num_corr, overlap_thresh=0.1):
        super(CoarseTargetGenerator, self).__init__()
        self.num_corr = num_corr
        self.overlap_thresh = overlap_thresh
    @jt.no_grad()
    def execute(self, gt_corr_indices, gt_corr_overlaps):
        gt_corr_masks = jt.greater(gt_corr_overlaps, self.overlap_thresh)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]
        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]
        if gt_corr_indices.shape[0] > self.num_corr:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_corr, replace=False)
            sel_indices = jt.Var(sel_indices)
            """ gt_ref_corr_indices = index_select(gt_ref_corr_indices, sel_indices, dim=0)
            gt_src_corr_indices = index_select(gt_src_corr_indices, sel_indices, dim=0)
            gt_corr_overlaps = index_select(gt_corr_overlaps, sel_indices, dim=0) """
            gt_ref_corr_indices = gt_ref_corr_indices[sel_indices.long()]
            gt_src_corr_indices = gt_src_corr_indices[sel_indices.long()]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices.long()]
        else:
            sel_indices=jt.zeros(gt_corr_indices.shape[0])

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps,sel_indices,gt_corr_masks