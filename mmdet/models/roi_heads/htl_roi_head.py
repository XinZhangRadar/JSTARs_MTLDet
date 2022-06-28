import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor,build_loss
from .cascade_roi_head import CascadeRoIHead
import pdb
from mmcv.cnn import kaiming_init
@HEADS.register_module()
class HybridTaskLearningCascadeRoIHead(CascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 denisity_roi_extractor=None,
                 denisity_head=None,
                 denisity_fusion=('bbox'),
                 edge_roi_extractor=None,
                 edge_head=None,
                 edge_fusion=('bbox'),
                 interleaved=True,
                 mask_info_flow=True,
                 loss_density=dict(
                    type='SSIM_Loss', in_channels=1, size=11, sigma=1.5, size_average=True),
                 loss_edge=dict(
                    type='WCE_Loss', w_positive = 1, w_negative = 1.1),
                 attention_fusion = True,
                 **kwargs):
        super(HybridTaskLearningCascadeRoIHead,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported
        #import pdb;pdb.set_trace()

        if denisity_head is not None:
            self.denisity_roi_extractor = build_roi_extractor(
                denisity_roi_extractor)
            self.denisity_head = build_head(denisity_head)
            self.with_denisity = True
        else:
            self.with_denisity = False
        if edge_head is not None:
            self.edge_roi_extractor = build_roi_extractor(
                edge_roi_extractor)
            self.edge_head = build_head(edge_head)
            self.with_edge = True
        
        else:
            self.with_edge = False

        self.denisity_fusion = denisity_fusion
        self.edge_fusion = edge_fusion        
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.loss_density = build_loss(loss_density)
        self.loss_edge = build_loss(loss_edge)
        self.attention_fusion = attention_fusion

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        #import pdb;pdb.set_trace()
        super(HybridTaskLearningCascadeRoIHead, self).init_weights(pretrained)
        if self.with_denisity:
            self.denisity_head.init_weights()
            input_ch = self.denisity_head.conv_out_channels
        if self.with_edge:
            self.edge_head.init_weights()
            input_ch = self.edge_head.conv_out_channels
        if self.with_denisity and self.attention_fusion :
            self.attention_conv_denisity = nn.Conv2d(input_ch, 1, 3, 1, 1)
            kaiming_init(self.attention_conv_denisity)
        if self.with_edge and self.attention_fusion :
            self.attention_conv_edge = nn.Conv2d(input_ch, 1, 3, 1, 1)
            kaiming_init(self.attention_conv_edge)
    @property

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        # semantic head
        if self.denisity:
            _, denisity_feat = self.denisity_head(x)
        else:
            denisity_feat = None
        if self.edge:
            _, edge_feat = self.edge_head(x)
        else:
            edge_feat = None

        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i, x, rois, denisity_feat=denisity_feat,edge_feat=edge_feat )
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            denisity_feat=None,
                            edge_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, denisity_feat=denisity_feat, edge_feat=edge_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'], rois,
                                   *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=bbox_targets,
        )
        return bbox_results



    def _bbox_forward(self, stage, x, rois, denisity_feat=None, edge_feat=None):
        """Box head forward function used in both training and testing."""
        #pdb.set_trace()
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_denisity and 'bbox' in self.denisity_fusion:
            #import pdb;pdb.set_trace()
            #import cv2
            #cv2.imwrite('den_feat.jpg',denisity_feat.mean(1).reshape(128,128).detach().cpu().numpy()*255)

            if self.attention_fusion:
                denisity_attention = torch.sigmoid(self.attention_conv_denisity(denisity_feat))
                denisity_feat = denisity_feat.mul(denisity_attention)
            bbox_denisity_feat = self.denisity_roi_extractor([denisity_feat],
                                                             rois)
            if bbox_denisity_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_denisity_feat = F.adaptive_avg_pool2d(
                    bbox_denisity_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_denisity_feat

        if self.with_edge and 'bbox' in self.edge_fusion:
            #import pdb;pdb.set_trace()
            if self.attention_fusion:
                edge_attention = torch.sigmoid(self.attention_conv_edge(edge_feat))
                edge_feat = edge_feat.mul(edge_attention)
            bbox_edge_feat = self.edge_roi_extractor([edge_feat],
                                                             rois)
            if bbox_edge_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_edge_feat = F.adaptive_avg_pool2d(
                    bbox_edge_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_edge_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results



    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_den=None,
                      gt_edge=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_den (None, Tensor) : true denisity heatmap 
                used if the architecture supports a segmentation task.

            gt_edge (None, list[Tensor]): edge masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        import pdb;pdb.set_trace()
        losses = dict()
        if self.with_denisity:
            denisity_pred, denisity_feat = self.denisity_head(x)
            #loss_denisity = self.denisity_head.loss(denisity_pred, gt_den)
            loss_denisity = self.loss_density(denisity_pred, gt_den.float())
            losses['loss_denisity'] = loss_denisity
        else:
            denisity_feat = None

        if self.with_edge:
            edge_pred, edge_feat = self.edge_head(x)
            loss_edge = self.loss_edge(edge_pred, gt_edge)
            losses['loss_edge'] = loss_edge
        else:
            edge_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, denisity_feat,edge_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

  
            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        #pdb.set_trace()
        if self.with_denisity:
            denisity_pred, denisity_feat = self.denisity_head(x)
        else:
            denisity_feat = None
            denisity_pred = None

        if self.with_edge:
            edge_pred, edge_feat = self.edge_head(x)
        else:
            edge_feat = None
            edge_pred = None

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_denisity_result = {}
        ms_edge_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        ms_denisity_result['ensemble'] = denisity_pred
        ms_edge_result['ensemble'] = edge_pred

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i, x, rois, denisity_feat=denisity_feat, edge_feat=edge_feat)
            ms_scores.append(bbox_results['cls_score'])

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label,
                                                  bbox_results['bbox_pred'],
                                                  img_metas[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        #results = (ms_bbox_result['ensemble'], ms_denisity_result['ensemble'], ms_edge_result['ensemble'])
        results = [ms_bbox_result['ensemble']]



        return results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
 
        if self.with_denisity:
            denisity_feats = [
                self.denisity_head(feat)[1] for feat in img_feats
            ]
        else:
            denisity_feats = [None] * len(img_metas)

        if self.with_edge:
            edge_feats = [
                self.edge_head(feat)[1] for feat in img_feats
            ]
        else:
            edge_feats = [None] * len(img_metas)


        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, denisity, edge in zip(img_feats, img_metas, denisity_feats,edge_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i, x, rois, denisity_feat=denisity, edge_feat=edge)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        
        return bbox_result
