import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
from PCR_Jittor.jittor.modules.loss import WeightedCircleLoss
from PCR_Jittor.jittor.modules.ops.transformation import apply_transform
from PCR_Jittor.jittor.modules.registration.metrics import isotropic_transform_error
from PCR_Jittor.jittor.modules.ops.pairwise_distance import pairwise_distance
import numpy as np
class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss( cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,)
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def execute(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = jt.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = jt.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = jt.greater(overlaps, self.positive_overlap)
        neg_masks = jt.equal(overlaps, 0)
        pos_scales = jt.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)
        if jt.isinf(loss):
            print('coarse')
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
    def __init__(self, config):
        super(FineMatchingLoss, self).__init__()
        self.pos_radius = config.fine_loss.positive_radius

    def execute(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points_ori = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transforms = data_dict['transform']

        transform=transforms[0]
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points_ori, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = jt.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = jt.less(dists, self.pos_radius ** 2)
        gt_corr_map = jt.logical_and(gt_corr_map, gt_masks)

        for i in range(len(transforms)-1):
            transform=transforms[i+1]
            src_node_corr_knn_points = apply_transform(src_node_corr_knn_points_ori, transform)
            dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
            gt_masks = jt.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
            gt_corr_map_tmp = jt.less(dists, self.pos_radius ** 2)
            gt_corr_map_tmp= jt.logical_and(gt_corr_map_tmp, gt_masks)
            gt_corr_map=jt.logical_or(gt_corr_map_tmp, gt_corr_map)
            
        slack_row_labels = jt.logical_and(jt.equal(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = jt.logical_and(jt.equal(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels =  jt.zeros_like(matching_scores, dtype='bool')
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()
        if jt.isnan(loss):
            print('fine')
            """ print(matching_scores)
            print(matching_scores[labels])
            print(matching_scores[labels].shape)
            print(ref_node_corr_knn_points.shape)
            print(src_node_corr_knn_points.shape)
            print(output_dict['ref_node_corr_knn_feats'])
            print(matching_scores.shape)
            print(gt_corr_map.shape)
            print(slack_row_labels.shape)
            print(slack_col_labels.shape) """

        return loss

class InstanceMaskLoss(nn.Module):
    def __init__(self, config):
        super(InstanceMaskLoss, self).__init__()

    
    def dice_loss(self,inputs,targets):

        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)  
        return loss.mean()

    def execute(self, output_dict):
        pred_masks_list=output_dict['pred_masks_list']
        gt_masks=output_dict['gt_masks']
        ref_node_corr_indices=output_dict['ref_node_corr_indices']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0].reshape(-1)
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        mask_bce_loss = jt.Var([0.0])
        mask_dice_loss = jt.Var([0.0])
        for pred_masks in pred_masks_list:
            mask_bce_loss += jt.nn.binary_cross_entropy_with_logits(pred_masks[gt_ref_node_corr_indices], gt_masks[gt_ref_node_corr_indices].float())
            mask_dice_loss += self.dice_loss(pred_masks[gt_ref_node_corr_indices],gt_masks[gt_ref_node_corr_indices].float())
        
        mask_bce_loss = mask_bce_loss / len(pred_masks_list)
        mask_dice_loss = mask_dice_loss / len(pred_masks_list)

        return mask_bce_loss,mask_dice_loss


class OverallLoss(nn.Module):
    def __init__(self, config):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(config)
        self.fine_loss = FineMatchingLoss(config)
        self.mask_loss=InstanceMaskLoss(config)
        self.weight_coarse_loss = config.loss.weight_coarse_loss
        self.weight_fine_loss = config.loss.weight_fine_loss
        self.weight_mask_loss = config.loss.weight_mask_loss

    def execute(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        mask_bce_loss,mask_dice_loss=self.mask_loss(output_dict)
        
        loss = self.weight_coarse_loss * coarse_loss +self.weight_fine_loss *fine_loss+self.weight_mask_loss*(mask_bce_loss+mask_dice_loss)
        result_dict = {
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'mask_bce_loss': mask_bce_loss,
            'mask_dice_loss':mask_dice_loss,
            'loss': loss
        }

        return result_dict


def  iou(box1,box2):
	'''
	3D IoU计算
	box表示形式：[x1,y1,z1,x2,y2,z2] 分别是两对角点的坐标
	'''
	in_w = min(box1[3],box2[3]) - max(box1[0],box2[0])
	in_l = min(box1[4],box2[4]) - max(box1[1],box2[1])
	in_h = min(box1[5],box2[5]) - max(box1[2],box2[2])

	inter = 0 if in_w < 0 or in_l < 0 or in_h < 0 else in_w * in_l * in_h
	union = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2]) + (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])  - inter
	iou = inter / union
	return iou




class Evaluator(nn.Module):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.positive_overlap = config.eval.acceptance_overlap
        self.positive_radius = config.eval.acceptance_radius
        self.re_thre=config.eval.rre_threshold
        self.te_thre=config.eval.rte_threshold
        self.add_s=0.1
        self.iou_ratio=0.7

    @jt.no_grad()
    def evaluate_coarse(self, output_dict, data_dict):
        ref_length_c = len(output_dict['ref_nodes'])
        src_length_c =  len(output_dict['src_nodes'])
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = jt.greater(gt_node_corr_overlaps, self.positive_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = jt.zeros(ref_length_c, src_length_c)
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @jt.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = jt.norm(ref_corr_points - src_corr_points, dim=1)
        precision = jt.less(corr_distances, self.positive_radius).float().mean()
        return precision
    
    


    @jt.no_grad()
    def evaluate_registrations(self, est_transforms, transforms):
        recall_trans = jt.zeros(len(transforms))
        precision_pred = jt.zeros(len(est_transforms))
        recall_trans_index = jt.zeros(len(transforms))-1
        precision_pred_index = jt.zeros(len(est_transforms))-1
        recall_best=jt.zeros((len(transforms),2))+361
        if len(est_transforms)>0:
            for i in range( len(transforms)):
                transform=transforms[i]
                for j in range(len(est_transforms)):
                    rre, rte = isotropic_transform_error(transform, est_transforms[j])
                    if rre < self.re_thre and rte < self.te_thre:
                        precision_pred[j] = 1
                        recall_trans[i] = 1
                        precision_pred_index[j] = i
                        if rre<=recall_best[i][0] and rte<=recall_best[i][1]:
                            recall_trans_index[i] = j
                            recall_best[i][0]=rre
                            recall_best[i][1]=rte
            precision = precision_pred.sum() / len(precision_pred)
            recall = recall_trans.sum() / len(recall_trans)

            return precision, recall,recall_trans_index,precision_pred_index,recall_best
        else:
            return 0.0,0.0,recall_trans_index,precision_pred_index,recall_best
    
    @jt.no_grad()
    def evaluate_sym_registrations(self, est_transforms, transforms,src_points):
        recall_trans = jt.zeros(len(transforms))
        precision_pred = jt.zeros(len(est_transforms))
        recall_trans_index = jt.zeros(len(transforms))-1
        precision_pred_index = jt.zeros(len(est_transforms))-1
        recall_best=jt.zeros((len(transforms),1))+361
        src_points_numpy=src_points.cpu().numpy()
        ab = np.matmul(src_points_numpy, src_points_numpy.transpose(-1, -2))
        a2 = np.expand_dims(np.sum(src_points_numpy ** 2, axis=-1), axis=-1)
        b2 = np.expand_dims(np.sum(src_points_numpy ** 2, axis=-1), axis=-2)
        dist2 = a2 - 2 * ab + b2
        src_R=np.sqrt(dist2.max())
        if len(est_transforms)>0:
            for i in range( len(transforms)):
                transform=transforms[i]
                for j in range(len(est_transforms)):       
                    src_pcd_gt = apply_transform(src_points, transform).cpu().numpy()
                    src_pcd_pred =apply_transform(src_points, est_transforms[j]).cpu().numpy()
                    ab = np.matmul(src_pcd_gt, src_pcd_pred.transpose(-1, -2))
                    a2 = np.expand_dims(np.sum(src_pcd_gt ** 2, axis=-1), axis=-1)
                    b2 = np.expand_dims(np.sum(src_pcd_pred ** 2, axis=-1), axis=-2)
                    dist2 = a2 - 2 * ab + b2
                    dist2 =dist2.min(1)
                    dist2=np.sqrt(dist2)
                    avg = np.average(dist2)
                    if avg<self.add_s*src_R:
                        precision_pred[j] = 1
                        recall_trans[i] = 1
                        precision_pred_index[j] = i
                        if avg/src_R<=recall_best[i][0]:
                            recall_trans_index[i] = j
                            recall_best[i][0]=avg/src_R
            precision = precision_pred.sum() / len(precision_pred)
            recall = recall_trans.sum() / len(recall_trans)

            return precision, recall,recall_trans_index,precision_pred_index,recall_best
        else:
            return 0.0,0.0,recall_trans_index,precision_pred_index,recall_best

    def execute(self, output_dict, data_dict):
        est_transforms = output_dict['estimated_transforms']
        transforms = data_dict['transform']
        if data_dict['sym']:
            precision, recall,recall_trans,precision_pred,recall_best = self.evaluate_registrations(est_transforms, transforms)
        else:
            precision, recall ,recall_trans,precision_pred,recall_best= self.evaluate_sym_registrations(est_transforms, transforms,output_dict['src_points_f'])
        if precision==0.0 and recall==0.0:
            F1_score=0.0
        else:
            F1_score=2 * (precision ) * (recall) / ((recall  +precision))

        result_dict = {
            'precision': precision,
            'recall': recall,
            'F1_score':F1_score,
            'recall_trans': recall_trans.long(),
            'recall_best':recall_best,
            'precision_pred': precision_pred.long(),
        }
        return result_dict
