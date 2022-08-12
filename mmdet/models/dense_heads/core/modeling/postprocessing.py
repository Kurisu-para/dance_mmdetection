# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
import functools
import multiprocessing as mp

from detectron2.structures import Instances, Boxes

import pycocotools.mask as mask_util

from core.structures import PolygonPoints
from core.utils import timer


def detector_postprocess(mask_result_src,
                         results,
                         output_height,
                         output_width,
                         re_comp_box):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # the results.image_size here is the one the model saw, typically (800, xxxx)

    # with timer.env('postprocess_sub1_get'):
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    # with timer.env('postprocess_sub2_scale'):
    output_boxes.scale(scale_x, scale_y)
        # now the results.image_size is the one of raw input image
    # with timer.env('postprocess_sub3_clip'):
    output_boxes.clip(results.image_size)

    # with timer.env('postprocess_sub4_filter'):
    results = results[output_boxes.nonempty()]

    # with timer.env('postprocess_cp2'):
    if results.has("pred_polys"):
        if results.has("pred_path"):
            with timer.env('extra'):
                snake_path = results.pred_path
                for i in range(snake_path.size(1)):     # number of evolution
                    current_poly = PolygonPoints(snake_path[:, i, :, :])
                    current_poly.scale(scale_x, scale_y)
                    current_poly.clip(results.image_size)
                    snake_path[:, i, :, :] = current_poly.tensor

        # TODO: Note that we did not scale exts (no need if not for evaluation)
        if results.has("ext_points"):
            results.ext_points.scale(scale_x, scale_y)

        results.pred_polys.scale(scale_x, scale_y)

        if re_comp_box:
            results.pred_boxes = Boxes(results.pred_polys.get_box())

        # results.pred_polys.clip(results.image_size)
        # results.pred_masks = get_polygon_rles(results.pred_polys.flatten(),
        #                                       (output_height, output_width))

        return results


    #elif results.has("ext_points"):
    # directly from extreme points to get these results as masks
    results.ext_points.scale(scale_x, scale_y)
    results.ext_points.fit_to_box()


    if mask_result_src == 'OCT_RLE':
        results.pred_masks = get_octagon_rles(results.ext_points.get_octagons(),
                                        (output_height, output_width))


    return results


def edge_map_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    # result = F.interpolate(
    #     result, size=(output_height, output_width), mode="bilinear", align_corners=False
    # )[0][0]
    return result[0][0]

def get_octagon_rles(octagons, image_shape):
    # input: N x 16
    octagons = octagons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([o.tolist()], h, w))
        for o in octagons
    ]
    return rles