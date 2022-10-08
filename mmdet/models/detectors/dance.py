# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet.core import bbox2result

from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class Dance(SingleStageInstanceSegmentor):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(Dance, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None,
                      positive_infos=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #gt_masks = [
        #    gt_mask.to_tensor(dtype=torch.bool, device=img.device)
        #    for gt_mask in gt_masks
        #]
        x = self.extract_feat(img)
        #这里的features层次选择，应该把参数加到什么地方比较合理？
        x_fcos = x[1:]
        x_edge = x[:4]
        losses = dict()
        bbox_head_preds = self.bbox_head(x_fcos)
        losses_fcos = self.bbox_head.loss(
                *bbox_head_preds,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
        #Some annotations can generate an empty TensorList sampled targets.
        losses.update(losses_fcos)
        #用mask_head表示edge部分
        losses_edge = self.mask_head.forward_train(x_edge, gt_bboxes, gt_masks,
                                            gt_semantic_seg, img_metas,gt_bboxes_ignore, positive_infos)
        losses.update(losses_edge)
        #用mask_head表示edge部分                                      
        #except RuntimeError:
        #    print("a non-empty poly_sample_location tensor is ignored here!")
        #    #just simply update losses_fcos again to make the length of loss the same as correct one
        #    losses_test = {}
        #    losses_test["loss_stage_0"] = losses_fcos['loss_cls']
        #    losses_test["loss_stage_1"] = losses_fcos['loss_bbox']
        #    losses_test["loss_stage_2"] = losses_fcos['loss_centerness']
        #    losses_test["loss_edge"] = losses_fcos['loss_ext']
        #    losses.update(losses_test)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        feat_fcos = feat[1:]
        feat_edge = feat[:4]
        result_list = self.bbox_head.simple_test(
            feat_fcos, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in result_list
        ]
        #in test, batch_size = 1
        if len(result_list) == 1:
            #no need for scores
            mask_pre = result_list[0][0]
            det_bboxes_mask_pre = mask_pre[:,:4]
            det_bboxes_mask = det_bboxes_mask_pre * det_bboxes_mask_pre.new_tensor(img_metas[0]['scale_factor'])
        else:
            det_bboxes_mask = [torch.cat(det_bboxes) for det_bboxes, _ in result_list]
        #it is not a good idea to adjust test batch size
        poly_list = self.mask_head.simple_test(feat_edge, det_bboxes_mask, img_metas)
        #poly_results_list = []
        #for poly in poly_list:
        #poly_results_list.append()
        poly_results = [self.format_poly_results(poly_list, det_labels, self.bbox_head.num_classes)
                        for _, det_labels in result_list]
        format_results_list = []
        for bbox_result, poly_result in zip(bbox_results, poly_results):
            format_results_list.append((bbox_result, poly_result))

        return format_results_list
        # bbox_results = [
        #    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #    for det_bboxes, det_labels in results_list
        #]
        # return bbox_results
    
    def format_poly_results(self, masks, labels, num_classes):
        num_masks = len(masks)
        mask_results = [[] for _ in range(num_classes)]
        if num_masks == 0:
            return mask_results
        else:
            labels = labels.detach().cpu().numpy()
            for idx in range(num_masks):
                mask = masks[idx]
                mask_results[labels[idx]].append(mask)
            return mask_results


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.
        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.
        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels