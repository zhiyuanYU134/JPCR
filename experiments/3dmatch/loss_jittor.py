import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
from PCR_Jittor.jittor.modules.loss import WeightedCircleLoss
from PCR_Jittor.jittor.modules.ops.transformation import apply_transform
from PCR_Jittor.jittor.modules.registration.metrics import isotropic_transform_error
from PCR_Jittor.jittor.modules.ops.pairwise_distance import pairwise_distance

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def execute(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0].reshape(-1)
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1].reshape(-1)

        feat_dists = jt.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = jt.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = jt.greater(overlaps, self.positive_overlap)
        neg_masks = jt.equal(overlaps, 0)
        pos_scales = jt.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)
        if jt.isinf(loss):
            print(feat_dists)
            print(gt_node_corr_overlaps)
            print(ref_feats.shape)
            print(src_feats.shape)
            print(feat_dists.shape)
            print(gt_ref_node_corr_indices.shape)
            print(gt_src_node_corr_indices.shape)
            print(pos_masks.shape)
            print(neg_masks.shape)
            print(pos_scales.shape)
            print(jt.isinf(feat_dists).any_())
            print(jt.isnan(feat_dists).any_())


        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def execute(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = jt.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = jt.less(dists, self.positive_radius ** 2)
        gt_corr_map = jt.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = jt.logical_and(jt.equal(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = jt.logical_and(jt.equal(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = jt.zeros_like(matching_scores, dtype='bool')
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()
        if jt.isnan(loss):
            print(matching_scores)
            print(matching_scores[labels])
            print(matching_scores[labels].shape)
            print(ref_node_corr_knn_points.shape)
            print(src_node_corr_knn_points.shape)
            print(ref_node_corr_knn_masks.shape)
            print(src_node_corr_knn_masks.shape)
            print(matching_scores.shape)
            print(gt_corr_map.shape)
            print(slack_row_labels.shape)
            print(slack_col_labels.shape)
            

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def execute(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @jt.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = jt.greater(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = jt.zeros(ref_length_c, src_length_c)
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @jt.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        print(ref_corr_points.shape)
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = jt.norm(ref_corr_points - src_corr_points, dim=1)
        precision = jt.less(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @jt.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = nn.matmul(linalg.inv(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse =jt.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = jt.less(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def execute(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
