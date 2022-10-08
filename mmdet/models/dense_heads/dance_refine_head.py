# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from typing import List, Union, Tuple
import pycocotools.mask as mask_util

from mmdet.core import InstanceData, mask_matrix_nms, multi_apply, BitmapMasks, PolygonMasks
from mmdet.core.utils import center_of_mass, generate_coordinate
from mmdet.models.builder import HEADS, build_loss
from .base_mask_head import BaseMaskHead
from mmcv.runner import force_fp32

from .core.layers import extreme_utils
from shapely.geometry import Polygon

class PolygonPoints:
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        :param tensor (Tensor[float]): a Nxkx2 tensor.  Last dim is (x, y);
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 128, 2, dtype=torch.float32, device=device)
        assert tensor.dim() == 3 and tensor.size(-1) == 2, tensor.size()

        self.tensor = tensor

    def clone(self) -> "PolygonPoints":

        return PolygonPoints(self.tensor.clone())

    def to(self, device: str) -> "PolygonPoints":
        return PolygonPoints(self.tensor.to(device))

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, :, 0] *= scale_x
        self.tensor[:, :, 1] *= scale_y

    def clip(self, box_size: BoxSizeType) -> None:
        assert torch.isfinite(self.tensor).all(), "Polygon tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[:, :, 0].clamp_(min=0, max=w)
        self.tensor[:, :, 1].clamp_(min=0, max=h)

    def flatten(self):
        n = self.tensor.size(0)
        if n == 0:
            return self.tensor
        return self.tensor.reshape(n, -1)

    def get_box(self):
        return torch.cat([self.tensor.min(1)[0], self.tensor.max(1)[0]], dim=1)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "PolygonPoints":
        """
        Returns:
            ExtremePoints: Create a new :class:`ExtremePoints` by indexing.

        The following usage are allowed:

        1. `new_exts = exts[3]`: return a `ExtremePoints` which contains only one box.
        2. `new_exts = exts[2:10]`: return a slice of extreme points.
        3. `new_exts = exts[vector]`, where vector is a torch.BoolTensor
           with `length = len(exts)`. Nonzero elements in the vector will be selected.

        Note that the returned ExtremePoints might share storage with this ExtremePoints,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return PolygonPoints(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 3, "Indexing on PolygonPoints with {} failed to return a matrix!".format(item)
        return PolygonPoints(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "PolyPts(" + str(self.tensor) + ")"

    @staticmethod
    def cat(pts_list: List["PolygonPoints"]) -> "PolygonPoints":
        """
        Concatenates a list of ExtremePoints into a single ExtremePoints

        Arguments:
            pts_list (list[PolygonPoints])

        Returns:
            pts: the concatenated PolygonPoints
        """
        assert isinstance(pts_list, (list, tuple))
        assert len(pts_list) > 0
        assert all(isinstance(pts, PolygonPoints) for pts in pts_list)

        cat_pts = type(pts_list[0])(torch.cat([p.tensor for p in pts_list], dim=0))
        return cat_pts

    @property
    def device(self) -> torch.device:
        return self.tensor.device

#from detectron2
def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_aux_extreme_points(pts):
    num_pt = pts.shape[0]

    aux_ext_pts = []

    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    band_thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    t_band = np.where((pts[:, 1] - t) <= band_thresh * h)[0].tolist()
    while t_band:
        t_idx = t_band[np.argmin(pts[t_band, 1])]
        t_idxs = [t_idx]
        tmp = (t_idx + 1) % num_pt
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (t_idx - 1) % num_pt
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        tt = (max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2
        aux_ext_pts.append(np.array([tt, t]))
        t_band = [item for item in t_band if item not in t_idxs]

    b_band = np.where((b - pts[:, 1]) <= band_thresh * h)[0].tolist()
    while b_band:
        b_idx = b_band[np.argmax(pts[b_band, 1])]
        b_idxs = [b_idx]
        tmp = (b_idx + 1) % num_pt
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (b_idx - 1) % num_pt
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        bb = (max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2
        aux_ext_pts.append(np.array([bb, b]))
        b_band = [item for item in b_band if item not in b_idxs]

    l_band = np.where((pts[:, 0] - l) <= band_thresh * w)[0].tolist()
    while l_band:
        l_idx = l_band[np.argmin(pts[l_band, 0])]
        l_idxs = [l_idx]
        tmp = (l_idx + 1) % num_pt
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (l_idx - 1) % num_pt
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        ll = (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2
        aux_ext_pts.append(np.array([l, ll]))
        l_band = [item for item in l_band if item not in l_idxs]

    r_band = np.where((r - pts[:, 0]) <= band_thresh * w)[0].tolist()
    while r_band:
        r_idx = r_band[np.argmax(pts[r_band, 0])]
        r_idxs = [r_idx]
        tmp = (r_idx + 1) % num_pt
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (r_idx - 1) % num_pt
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        rr = (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2
        aux_ext_pts.append(np.array([r, rr]))
        r_band = [item for item in r_band if item not in r_idxs]

    # assert len(aux_ext_pts) >= 4
    pt0 = aux_ext_pts[0]

    # collecting
    aux_ext_pts = np.stack(aux_ext_pts, axis=0)

    # ordering
    shift_idx = np.argmin(np.power(pts - pt0, 2).sum(axis=1))
    re_ordered_pts = np.roll(pts, -shift_idx, axis=0)

    # indexing
    ext_idxs = np.argmin(np.sum(
        (aux_ext_pts[:, np.newaxis, :] - re_ordered_pts[np.newaxis, ...]) ** 2, axis=2),
        axis=1)
    ext_idxs[0] = 0

    ext_idxs = np.sort(np.unique(ext_idxs))

    return re_ordered_pts, ext_idxs

#In fact, batch size is only set as 1 in test
#transform the masks into bitmapmasks, which fit the evaluation procedure
def get_polygon_rles(polygons, image_shape):
    # input: N x (p*2)
    polygons = polygons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([p.tolist()], h, w)) for p in polygons
    ]
    bitmap_masks = [mask_util.decode(rle).astype(np.bool)
                    for rle in rles]
    return bitmap_masks

class DilatedCircularConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircularConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(
            state_dim,
            out_state_dim,
            kernel_size=self.n_adj * 2 + 1,
            dilation=self.dilation,
        )

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [
                    input[..., -self.n_adj * self.dilation :],
                    input,
                    input[..., : self.n_adj * self.dilation],
                ],
                dim=2,
            )
        return self.fc(input)

class SnakeBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(SnakeBlock, self).__init__()

        self.conv = DilatedCircularConv(state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)

        return x

class _SnakeNet(nn.Module):
    #stage_num = 3,  实际不需要这个参数，在默认config设定中每一层的dilation等是一致的
    def __init__(self, state_dim, feature_dim):
        super(_SnakeNet, self).__init__()

        self.state_dim = state_dim   #128
        # why +2?
        self.feature_dim = feature_dim + 2   #256+2

        self.head = SnakeBlock(self.feature_dim, self.state_dim)

        self.res_layer_num = 7   #(8 - 1) * 3
        dilation = [1, 1, 1, 2, 2, 4, 4]   #(1, 1, 1, 2, 2, 4, 4) * 3
        for i in range(self.res_layer_num):
            conv = SnakeBlock(self.state_dim, self.state_dim, n_adj=4, dilation=dilation[i])
            self.__setattr__("res" + str(i), conv)

        fusion_state_dim = 256

        # if self.skip:
        #     fusion_state_dim = feature_dim

        self.fusion = nn.Conv1d(
            self.state_dim * (self.res_layer_num + 1), fusion_state_dim, 1
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(self.state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1),
        )

    def forward(self, x):
        states = []

        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        back_out = self.fusion(state)
        global_state = torch.max(back_out, dim=2, keepdim=True)[0]

        # # big skip to conn spatial feat from 2D conv.
        # if self.skip:
        #     back_out += x

        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x

@HEADS.register_module()
class RefineHead(BaseMaskHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        state_dim = 128,
        feat_channels = 256,
        strides=(8, 16, 32, 64),
        num_iter = (0, 0, 1),
        num_convs = 2,
        num_sampling = 196,
        in_features = ['p2', 'p3', 'p4', 'p5'],
        common_stride=4,
        loss_refine=dict(
                    type='SmoothL1Loss', loss_weight=10.0),
        loss_edge=dict(type='DiceIgnoreLoss', use_sigmoid = False, activate = False, loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        train_cfg=None,
        test_cfg=None,
        init_cfg=[dict(type='Kaiming', layer='Conv2d'),
                  dict(type='Normal', layer='Conv1d', std=0.01,
                        override=[dict(type='Normal', name='snake_conv', std=0.01),
                                  dict(type='Normal', name='conv_att', std=0.01)])]
    ):
        super(RefineHead, self).__init__(init_cfg)
        self.common_stride = common_stride
        self.in_features = in_features
        self.num_classes = num_classes
        #self.cls_out_channels = self.num_classes
        self.in_channels = in_channels   #256
        self.state_dim = state_dim   #128
        self.feat_channels = feat_channels   #256
        self.strides = strides
        self.num_iter = num_iter   #(0, 0, 1)  correspond to the convs.  It seems the value in it is not important at all//
        self.num_convs = num_convs   #2
        self.num_sampling = num_sampling   #196
        # number of FPN feats
        self.num_levels = len(strides)
        self.loss_refine = build_loss(loss_refine)
        self.loss_edge = build_loss(loss_edge)
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        #init RefineNet/Snake head
        self._init_att_convs()
        self._init_predictor_att()
        self._init_edge_convs()
        self._init_snake_convs()
    
    #snake_conv == bottom_out
    def _init_snake_convs(self):
        self.snake_conv = nn.ModuleList()
        for i in range(self.num_convs):
            self.snake_conv.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    #bias='auto'
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU')
            )
        )

        # snakes
        for i in range(len(self.num_iter)):
            #self.conv_type == "ccn"
            snake_deformer = _SnakeNet(self.state_dim, self.feat_channels)   
            self.__setattr__("deformer" + str(i), snake_deformer)

        #initialization
        #for m in self.modules():
        #    if (
        #        isinstance(m, nn.Conv2d)
        #        or isinstance(m, nn.Conv1d)
        #    ):
        #        nn.init.normal_(m.weight, 0.0, 0.01)
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)

    #组装attender
    def _init_att_convs(self):
        self.att_convs = nn.ModuleList()
        self.att_convs.append(
            ConvModule(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                #bias='auto'
                norm_cfg=dict(type='GN', num_groups=4),
                act_cfg=dict(type='ReLU')
            )
        )
        self.att_convs.append(
            ConvModule(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                #bias='auto'
                norm_cfg=dict(type='GN', num_groups=4),
                act_cfg=dict(type='ReLU')
            )
        )
    
    def _init_predictor_att(self):
        self.conv_att = nn.Conv2d(
            32, 1, kernel_size=3, stride=1, padding=1, bias=True
        )
        #nn.init.normal_(self.conv_att.weight, 0, 0.01)
        #nn.init.constant_(self.conv_att.bias, 0)
        self.conv_att_activate = nn.Sigmoid()

    #组装 edge_predictor  
    def _init_edge_convs(self):
        #if self.edge_on
        self.scale_heads = []
        #self.in_features:  p2,p3,p4,p5
        #对于mmdet来说，input应该是fpn的output,也即x
        for i, in_feature in enumerate(self.in_features):
            head_ops = []
            head_length = max(
                1,
                int(
                    np.log2(self.strides[i])
                    - np.log2(self.common_stride)
                ),
            )
            for k in range(head_length):
                conv = ConvModule(
                    self.in_channels if k == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    #bias='auto',
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU')
                )
                head_ops.append(conv)
                if self.strides[i] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        #对应predictor
        self.pred_convs = nn.ModuleList()
        self.pred_convs.append(
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                #bias='auto'
                norm_cfg=dict(type='GN', num_groups=32),
            ) 
        )
        #num_classes = 1   class agnostic   
        self.pred_convs.append(
            ConvModule(
                self.feat_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                #bias='auto'
                act_cfg=None,
            )
        )

    @staticmethod    
    def uniform_sample(pgtnp_px2, newpnum):
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum :]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            # for i in range(pnum):
            #     if edgenum[i] == 0:
            #         edgenum[i] = 1
            edgenum[edgenum == 0] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:
                if edgenumsum > newpnum:
                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        # take the longest and divide.
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum  # terminate
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i : i + 1]
                pe_1x2 = pgtnext_px2[i : i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = (
                    np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]
                )

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    @staticmethod
    def uniform_sample_1d(pts, new_n):
        if new_n == 1:
            return pts[:1]
        n = pts.shape[0]
        if n == new_n + 1:
            return pts[:-1]
        # len: n - 1
        segment_len = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))

        # down-sample or up-sample
        # n
        start_node = np.cumsum(np.concatenate([np.array([0]), segment_len]))
        total_len = np.sum(segment_len)

        new_per_len = total_len / new_n

        mark_1d = ((np.arange(new_n - 1) + 1) * new_per_len).reshape(-1, 1)
        locate = start_node.reshape(1, -1) - mark_1d
        iss, jss = np.where(locate > 0)
        cut_idx = np.cumsum(np.unique(iss, return_counts=True)[1])
        cut_idx = np.concatenate([np.array([0]), cut_idx[:-1]])

        after_idx = jss[cut_idx]
        before_idx = after_idx - 1

        after_idx[after_idx < 0] = 0

        before = locate[np.arange(new_n - 1), before_idx]
        after = locate[np.arange(new_n - 1), after_idx]

        w = (-before / (after - before)).reshape(-1, 1)

        sampled_pts = (1 - w) * pts[before_idx] + w * pts[after_idx]

        return np.concatenate([pts[:1], sampled_pts], axis=0)

    @staticmethod
    def uniform_upsample(poly, p_num):
        if poly.size(1) == 0:
            return torch.zeros([0, p_num, 2], device=poly.device), None

        # 1. assign point number for each edge
        # 2. calculate the coefficient for linear interpolation
        next_poly = torch.roll(poly, -1, 2)
        edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
        edge_num = torch.round(
            edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]
        ).long()
        edge_num = torch.clamp(edge_num, min=1)

        edge_num_sum = torch.sum(edge_num, dim=2)
        edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
        extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
        edge_num_sum = torch.sum(edge_num, dim=2)
        assert torch.all(edge_num_sum == p_num)

        edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
        weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
        poly1 = poly.gather(
            2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2)
        )
        poly2 = poly.gather(
            2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2)
        )
        poly = poly1 * (1 - weight) + poly2 * weight

        return poly[0], edge_start_idx[0]

    #pred_instance
    def single_sample_bboxes_fast(self, pred_instances):
        instance_per_im = pred_instances
        xmin, ymin = (
            instance_per_im[:, 0],
            instance_per_im[:, 1],
        )  # (n,)
        xmax, ymax = (
            instance_per_im[:, 2],
            instance_per_im[:, 3],
        )  # (n,)
        box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
        box = torch.stack(box, dim=1).view(-1, 4, 2)
        sampled_box, _ = self.uniform_upsample(box[None], self.num_sampling)
        return sampled_box, None

    @staticmethod
    def get_simple_contour(gt_masks):
        polygon_mask = gt_masks
        contours = []

        for polys in polygon_mask:
            # polys = binary_mask_to_polygon(mask)
            contour = list(
                map(lambda x: np.array(x).reshape(-1, 2).astype(np.float32), polys)
            )
            if len(contour) > 1:  # ignore fragmented instances
                contours.append(None)
            else:
                contours.append(contour[0])
        return contours

    #对Polygon类的调用主要在这个方法里面
    #https://github.com/shapely/shapely/blob/main/shapely/geometry/polygon.py
    def compute_targets_for_polys(self, gt_bboxes, gt_masks, image_sizes):
        poly_sample_locations = []
        poly_sample_targets = []
        # dense_sample_targets = []  # 3x number of sampling
        # init_box_locs = []
        # init_ex_targets = []
        image_index = []
        # scales = []
        # cls = []
        whs = []

        #if self.new_matching:
        up_rate = 5  # TODO: subject to change, (hard-code 5x.
        #else:
        #    up_rate = 1

        # per image
        for im_i in range(len(image_sizes)):
            img_size_per_im = image_sizes[im_i]
            bboxes = gt_bboxes[im_i]
            # classes = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                continue

            cur_masks = gt_masks[im_i].masks
            # use this as a scaling
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]            
                #if (self.initial == "box") and (not self.original):
                # upper_right = torch.stack([bboxes[:, None, 2], bboxes[:, None, 1]], dim=2)
                # upper_left = torch.stack([bboxes[:, None, 0], bboxes[:, None, 1]], dim=2)
                # bottom_left = torch.stack([bboxes[:, None, 0], bboxes[:, None, 3]], dim=2)
                # bottom_right = torch.stack([bboxes[:, None, 2], bboxes[:, None, 3]], dim=2)
                # octagons = torch.cat([upper_right, upper_left, bottom_left, bottom_right], dim=1)
                # print('wrong')
            xmin, ymin = bboxes[:, 0], bboxes[:, 1]  # (n,)
            xmax, ymax = bboxes[:, 2], bboxes[:, 3]  # (n,)
            box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
            box = torch.stack(box, dim=1).view(-1, 4, 2)
            # uniformly sample on the box contour, set points based on the length of edge
            octagons, _ = self.uniform_upsample(
                box[None], self.num_sampling
            )

            # just to suppress errors (DUMMY):
            init_box, _ = self.uniform_upsample(box[None], 40)
            # ex_pts = init_box

            # List[np.array], element shape: (P, 2) OR None
            # change the mask form into (x,y) coordinate. Note that if instance is fragmented, it will be ignored.
            contours = self.get_simple_contour(cur_masks)

            # per instance
            # for (oct, cnt, w, h) in zip(octagons, contours, ws, hs):
            for (oct, cnt, in_box, w, h) in zip(
                octagons, contours, init_box, ws, hs
            ):
                if cnt is None:
                    continue

                # debug: stack non-empty Tensor
                # used for normalization
                # scale = torch.min(w, h)

                # make it clock-wise
                cnt = cnt[::-1] if Polygon(cnt).exterior.is_ccw else cnt
                """
                Quick fix for cityscapes
                """
                if Polygon(cnt).exterior.is_ccw:
                    continue

                assert not Polygon(
                    cnt
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                oct_sampled_pts = oct.cpu().numpy()
                assert not Polygon(
                    oct_sampled_pts
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                to_check = in_box.cpu().numpy()
                assert not Polygon(
                    to_check
                ).exterior.is_ccw, "0) init box must be clock-wise!"

                # sampling from ground truth
                # sample from contour(gt masks)
                oct_sampled_targets = self.uniform_sample(
                    cnt, len(cnt) * self.num_sampling * up_rate
                )  # (big, 2)
                # oct_sampled_targets = self.uniform_sample(cnt, len(cnt) * self.num_sampling * up_sample_rate)
                # i) find a single nearest, so that becomes ordered point sets
                # tt_idx = np.argmin(
                #     np.power(oct_sampled_targets - oct_sampled_pts[0], 2).sum(axis=1)
                # )
                # oct_sampled_targets = np.roll(oct_sampled_targets, -tt_idx, axis=0)[
                #     :: len(cnt)
                # ]

                # if self.initial == "box" and self.new_matching:
                oct_sampled_targets, aux_ext_idxs = get_aux_extreme_points(
                    oct_sampled_targets
                )
                tt_idx = np.argmin(
                    np.power(oct_sampled_pts - oct_sampled_targets[0], 2).sum(
                        axis=1
                    )
                )
                oct_sampled_pts = np.roll(oct_sampled_pts, -tt_idx, axis=0)
                oct = torch.from_numpy(oct_sampled_pts).to(oct.device)
                oct_sampled_targets = self.single_uniform_multisegment_matching(
                    oct_sampled_targets, oct_sampled_pts, aux_ext_idxs, up_rate
                )
                oct_sampled_targets = torch.tensor(
                    oct_sampled_targets, device=bboxes.device
                ).float()
                # else:
                #    oct_sampled_targets = torch.tensor(
                #        oct_sampled_targets, device=bboxes.device
                #    )
                # assert not Polygon(oct_sampled_targets).exterior.is_ccw, '2) contour must be clock-wise!'

                # oct_sampled_pts = torch.tensor(oct_sampled_pts, device=bboxes.device)
                # dense_targets = torch.tensor(dense_targets, device=bboxes.device)

                oct_sampled_targets[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                oct_sampled_targets[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)

                # dense_targets = oct_sampled_targets

                # Jittering should happen after all the matching
                """if self.jittering:
                    turbulence = (
                        torch.randn_like(ex_pts, device=bboxes.device) * self.jittering
                    )

                    # for box
                    ex_pts[:, 1::2, 0] += turbulence[:, 1::2, 0] * ws[:, None]
                    ex_pts[:, 0::2, 1] += turbulence[:, 0::2, 1] * hs[:, None]
                    # for ext
                    ex_pts[:, 0::2, 0] += turbulence[:, 0::2, 0] * ws[:, None] * 0.25
                    ex_pts[:, 1::2, 1] += turbulence[:, 1::2, 1] * hs[:, None] * 0.25

                    ex_pts[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                    ex_pts[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)"""

                poly_sample_locations.append(oct)
                # dense_sample_targets.append(dense_targets)
                poly_sample_targets.append(oct_sampled_targets)
                image_index.append(im_i)
                # scales.append(scale)
                whs.append([w, h])
                # init_box_locs.append(in_box)
                # init_ex_targets.append(ex_tar)

        # init_ex_targets = torch.stack(init_ex_targets, dim=0)
        if len(poly_sample_locations) == 0:
            print(poly_sample_locations)
            cur_masks_tensor = torch.tensor(cur_masks * 0).to(bboxes.device)
            poly_sample_locations = torch.sum(bboxes) * 0
            poly_sample_targets = torch.sum(cur_masks_tensor) * 0
            # scales = torch.zeros(1).to(bboxes.device)
            err_flag = True
            print('Stack expects a non-empty tensor!!')
        else:
            poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
            poly_sample_targets = torch.stack(poly_sample_targets, dim=0)
            # scales = torch.stack(scales, dim=0)
            err_flag = False
        # init_box_locs = torch.stack(init_box_locs, dim=0)
        # init_ex_targets = torch.stack(init_ex_targets, dim=0)
        # dense_sample_targets = torch.stack(dense_sample_targets, dim=0)
        # edge_index = torch.stack(edge_index, dim=0)
        image_index = torch.tensor(image_index, device=bboxes.device)
        whs = torch.tensor(whs, device=bboxes.device)

        # cls = torch.stack(cls, dim=0)
        return {
            "sample_locs": poly_sample_locations,
            "sample_targets": poly_sample_targets,
            # "sample_dense_targets": dense_sample_targets,
            # "scales": scales,
            "whs": whs,
            # "edge_idx": edge_index,
            "image_idx": image_index,
            "err_flag": err_flag,
            # "init_locs": init_box_locs,
            # "init_targets": init_ex_targets,
        }

    def single_uniform_multisegment_matching(
        self, dense_targets, sampled_pts, ext_idx, up_rate
    ):
        """
        Several points to note while debugging:
        1) For GT (from which points are sampled), include both end by [s, e + 1] indexing.
        2) If GT not increasing (equal), shift forwards by 1.
        3) Check the 1st sampled point is indexed by 0.
        4) Check the last sampled point is NOT indexed by 0 or any small value.
        """
        min_idx = ext_idx

        ch_pts = dense_targets[min_idx]  # characteristic points

        diffs = ((ch_pts[:, np.newaxis, :] - sampled_pts[np.newaxis]) ** 2).sum(axis=2)
        ext_idx = np.argmin(diffs, axis=1)
        if ext_idx[0] != 0:
            ext_idx[0] = 0
        if ext_idx[-1] < ext_idx[1]:
            ext_idx[-1] = self.num_sampling - 2
        ext_idx = np.sort(ext_idx)

        aug_ext_idx = np.concatenate([ext_idx, np.array([self.num_sampling])], axis=0)

        # diff = np.sum((ch_pts[:, np.newaxis, :] - dense_targets[np.newaxis, :, :]) ** 2, axis=2)
        # min_idx = np.argmin(diff, axis=1)

        aug_min_idx = np.concatenate(
            [min_idx, np.array([self.num_sampling * up_rate])], axis=0
        )

        if aug_min_idx[-1] < aug_min_idx[1]:
            aug_min_idx[-1] = (
                self.num_sampling * up_rate - 2
            )  # enforce matching of the last point

        if aug_min_idx[0] != 0:
            # TODO: This is crucial, or other wise the first point may be
            # TODO: matched to near 640, then sorting will completely mess
            aug_min_idx[0] = 0  # enforce matching of the 1st point

        aug_ext_idx = np.sort(aug_ext_idx)
        aug_min_idx = np.sort(aug_min_idx)

        # === error-prone ===

        # deal with corner cases

        if aug_min_idx[-2] == self.num_sampling * up_rate - 1:
            # print("WARNING: Bottom extreme point being the last point!")
            #self._logger.info("WARNING: Bottom extreme point being the last point!")
            # hand designed remedy
            aug_min_idx[-2] = self.num_sampling * up_rate - 3
            aug_min_idx[-1] = self.num_sampling * up_rate - 2

        if aug_min_idx[-1] == self.num_sampling * up_rate - 1:
            # print("WARNING: Right extreme point being the last point!")
            #self._logger.info("WARNING: Right extreme point being the last point!")
            #self._logger.info(aug_ext_idx)
            #self._logger.info(aug_min_idx)
            aug_min_idx[-1] -= 1
            aug_min_idx[-2] -= 1

        segments = []
        try:
            for i in range(len(ext_idx)):
                if aug_ext_idx[i + 1] - aug_ext_idx[i] == 0:
                    continue  # no need to sample for this segment

                if aug_min_idx[i + 1] - aug_min_idx[i] <= 0:
                    # overlap due to quantization, negative value is due to accumulation of overlap
                    aug_min_idx[i + 1] = aug_min_idx[i] + 1  # guarantee spacing

                if i == len(ext_idx) - 1:  # last, complete a circle
                    pts = np.concatenate(
                        [dense_targets[aug_min_idx[i] :], dense_targets[:1]], axis=0
                    )
                else:
                    pts = dense_targets[
                        aug_min_idx[i] : aug_min_idx[i + 1] + 1
                    ]  # including
                new_sampled_pts = self.uniform_sample_1d(
                    pts, aug_ext_idx[i + 1] - aug_ext_idx[i]
                )
                segments.append(new_sampled_pts)
            # segments.append(dense_targets[-1:]) # close the loop
            segments = np.concatenate(segments, axis=0)
            if len(segments) != self.num_sampling:
                # print("WARNING: Number of points not matching!")
                #self._logger.info(
                #    "WARNING: Number of points not matching!", len(segments)
                #)
                raise ValueError(len(segments))
        except Exception as err:  # may exist some very tricky corner cases...
            # print("WARNING: Tricky corner cases occurred!")
            #self._logger.info("WARNING: Tricky corner cases occurred!")
            #self._logger.info(err)
            #self._logger.info(aug_ext_idx)
            #self._logger.info(aug_min_idx)
            # raise ValueError('TAT')
            segments = self.reorder_perloss(
                torch.from_numpy(dense_targets[::up_rate][None]),
                torch.from_numpy(sampled_pts)[None],
            )[0]
            segments = segments.numpy()

        return segments

    def reorder_perloss(self, oct_sampled_targets, oct_sampled_pts):
        """
        Adaptively adjust the penalty, concept-wise the loss is much more reasonable.
        :param oct_sampled_targets: (\sum{k}, num_sampling, 2) for all instances
        :param oct_sampled_pts: same~
        :return:
        """
        assert oct_sampled_targets.size() == oct_sampled_pts.size()
        n = len(oct_sampled_targets)
        num_locs = oct_sampled_pts.size(1)
        ind1 = torch.arange(num_locs, device=oct_sampled_targets.device)
        ind2 = ind1.expand(num_locs, -1)
        enumerated_ind = torch.fmod(ind2 + ind1.view(-1, 1), num_locs).view(-1).long()
        enumerated_targets = oct_sampled_targets[:, enumerated_ind, :].view(
            n, -1, num_locs, 2
        )
        diffs = enumerated_targets - oct_sampled_pts[:, None, ...]
        diffs_sum = diffs.pow(2).sum(3).sum(2)
        tt_idx = torch.argmin(diffs_sum, dim=1)
        re_ordered_gt = enumerated_targets[torch.arange(n), tt_idx]
        return re_ordered_gt

    def get_locations_feature(self, features, locations, image_idx):
        h = features.shape[2] * 4
        w = features.shape[3] * 4
        locations = locations.clone()
        locations[..., 0] = locations[..., 0] / (w / 2.0) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.0) - 1

        # if (locations > 1).any() or (locations < -1).any():
        #     print('exceed grid sample boundary')
        #     if (locations > 1).any():
        #         print(locations[torch.where(locations>1)])
        #     else:
        #         print(locations[torch.where(locations < -1)])

        batch_size = features.size(0)
        sampled_features = torch.zeros(
            [locations.size(0), features.size(1), locations.size(1)],
            device=locations.device,
        )
        for i in range(batch_size):
            if image_idx is None:
                per_im_loc = locations.unsqueeze(0)
            else:
                per_im_loc = locations[image_idx == i].unsqueeze(0)
            # TODO: After all almost fixed, try padding_mode='reflection' to see if there's improv
            feature = torch.nn.functional.grid_sample(
                features[i : i + 1],
                per_im_loc,
                padding_mode="reflection",
                align_corners=False,
            )[0].permute(1, 0, 2)
            if image_idx is None:
                sampled_features = feature
            else:
                sampled_features[image_idx == i] = feature

        return sampled_features

    def de_location(self, locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        x_max = torch.max(locations[..., 0], dim=-1)[0]
        y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()
        #if self.de_location_type == "derange":  # [0, 1]
        new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None]) / (
            x_max[..., None] - x_min[..., None]
        )
        new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None]) / (
            y_max[..., None] - y_min[..., None]
        )

        #elif self.de_location_type == "demean":  # [-1, 1]
        #    new_locations[..., 0] = (
        #        2.0
        #        * (new_locations[..., 0] - x_min[..., None])
        #        / (x_max[..., None] - x_min[..., None])
        #        - 1.0
        #    )
        #    new_locations[..., 1] = (
        #        2.0
        #        * (new_locations[..., 1] - y_min[..., None])
        #        / (y_max[..., None] - y_min[..., None])
        #        - 1.0
        #    )
        #elif self.de_location_type == "demin":
        #    new_locations[..., 0] = new_locations[..., 0] - x_min[..., None]
        #    new_locations[..., 1] = new_locations[..., 1] - y_min[..., None]
        #else:
        #    raise ValueError("Invalid operation!", self.de_location_type)

        return new_locations

    def evolve(
        self,
        deformer,
        features,
        locations,
        image_idx,
        image_sizes,
        whs,
        att=False,
        path=None,
    ):

        locations_for_sample = locations.detach()

        # with timer.env('snake_sample_feat'):
        sampled_features = self.get_locations_feature(
            features, locations_for_sample, image_idx
        )

        #if self.attention:
        att_scores = sampled_features[:, :1, :]
        sampled_features = sampled_features[:, 1:, :]

        calibrated_locations = self.de_location(locations_for_sample)
        concat_features = torch.cat(
            [sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1
        )

        #self.conv_type == "ccn":
        pred_offsets = deformer(concat_features)      
        pred_offsets = pred_offsets.permute(0, 2, 1)

        #if self.individual_scale:
        pred_offsets = torch.tanh(pred_offsets) * whs[:, None, :]

        if att:
            # print('att scores', att_scores)
            pred_offsets = pred_offsets * att_scores.permute(0, 2, 1)

        pred_locations = locations + pred_offsets

        self.clip_locations(pred_locations, image_idx, image_sizes)

        return pred_locations

    @staticmethod
    def clip_locations(pred_locs, image_idx, image_sizes):
        if image_idx is None:
            pred_locs[0, :, 0::2].clamp_(min=0, max=image_sizes[0][1] - 1)
            pred_locs[0, :, 1::2].clamp_(min=0, max=image_sizes[0][0] - 1)
        else:
            for i, img_size_per_im in enumerate(image_sizes):
                pred_locs[image_idx == i, :, 0::2].clamp_(
                    min=0, max=img_size_per_im[1] - 1
                )
                pred_locs[image_idx == i, :, 1::2].clamp_(
                    min=0, max=img_size_per_im[0] - 1
                )

    def forward(self, feats):
        #每一层相对应的过一个scale_heads
        for i in range(len(self.in_features)):
            if i == 0:
                x = self.scale_heads[i](feats[i])
            else:
                x = x + self.scale_heads[i](feats[i])
        input_feat = x
        #convolution for nn.ModuleList
        #generate predictions
        for pred_layer in (self.pred_convs):
            x = pred_layer(x)
        pred_edge = x.sigmoid()
        #attentnion procedure
        att_temp = 1 - pred_edge
        #att_temp = 1 - x
        for att_layer in (self.att_convs):
            att_temp = att_layer(att_temp)  # regions that need evolution
        att_map = self.conv_att(att_temp)
        att_map_acted = self.conv_att_activate(att_map)

        #Snake head input features
        snake_input = torch.cat([att_map_acted, input_feat], dim=1)

        seg_preds = F.interpolate(
            pred_edge,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        
        edge_band = snake_input[:, :1, ...]
        features = snake_input[:, 1:, ...]

        for i in range(self.num_convs):
            features = self.snake_conv[i](features)

        #if self.attention:
        features = torch.cat([edge_band, features], dim=1)
        #pred_instances即box_list

        return seg_preds, features
        
        #if not self.training
        #return pred_edge, {}, new_instances    
    
    #@force_fp32(apply_to=('seg_preds', ))
    def loss(self, features, seg_preds, gt_bboxes, gt_masks, gt_semantic_seg, img_metas, **kwargs):
        loss = {}
        location_preds = []
        image_sizes = []
        for im_i in range(len(img_metas)):
            image_sizes.append(img_metas[im_i]['img_shape'][:2])
        training_targets = self.compute_targets_for_polys(gt_bboxes, gt_masks, image_sizes)

        locations, reg_targets, image_idx, whs, err_flag = (
            training_targets["sample_locs"],
            training_targets["sample_targets"],
            training_targets["image_idx"],
            training_targets["whs"],
            training_targets["err_flag"]
            )
        # location = ([])
        #vis(image, locations, reg_targets)
        edge_loss = self.loss_edge(seg_preds, gt_semantic_seg)
        if err_flag:
            loss["loss_stage_0"] = (torch.sum(features) * 0).to(device=edge_loss.device)
            loss["loss_stage_1"] = locations.to(device=edge_loss.device)
            loss["loss_stage_2"] = reg_targets.to(device=edge_loss.device)
            print(loss)
        else:
            for i in range(len(self.num_iter)):
                deformer = self.__getattr__("deformer" + str(i))
                if i == 0:
                    pred_location = self.evolve(
                        deformer,
                        features,
                        locations,
                        image_idx,
                        image_sizes,
                        whs,
                    )
                else:
                    pred_location = self.evolve(
                        deformer,
                        features,
                        pred_location,
                        image_idx,
                        image_sizes,
                        whs,
                        att=True,
                    )
                #vis(image, pred_location.detach(), reg_targets)
                location_preds.append(pred_location)
            for i, (pred) in enumerate(location_preds):
                loss_name = "loss_stage_" + str(i)
                stage_weight = 1 / 3
                #if not self.loss_adaptive:
                #dynamic_reg_targets = reg_targets

                #if not self.point_loss_weight:
                #point_weight = (
                #    torch.tensor(1, device=scales.device).float()
                #    / scales[:, None, None]
                #)
                #if self.individual_scale:
                #相当于进行一个分别的归一化
                #test 不做归一化
                point_weight = (
                    torch.tensor(1, device=whs.device).float()
                    / whs[:, None, :]
                )

                stage_loss = (
                    self.loss_refine(
                        pred * point_weight, reg_targets * point_weight
                    )
                    #self.loss_refine(
                    #    pred, reg_targets
                    #)
                    * stage_weight
                )

                loss[loss_name] = stage_loss
        loss.update(dict(loss_edge = edge_loss))
        return loss

    def get_results(self, features, proposals, img_metas):
        location_preds = []
        locations, image_idx = self.single_sample_bboxes_fast(proposals)

        if len(locations) == 0:
            return proposals
        image_sizes = []
        image_sizes_ori = []
        scale_factors = []
        for im_i in range(len(img_metas)):
            image_sizes.append(img_metas[im_i]['img_shape'][:2])
            image_sizes_ori.append(img_metas[im_i]['ori_shape'][:2])
            #warning: can scale factor be diffierent in some occasions?
            scale_factors.append(img_metas[im_i]['scale_factor'][:2])
        #img_shape: (h, w, 3)
        # print(image_sizes)
        # print(features.shape)
        # bboxes = pred_instances[0].pred_boxes.tensor
        # print(bboxes.max(0)[0], bboxes.min(0)[0])
        # bboxes = proposals
        #if unecessary?
        bboxes = proposals

        ws = bboxes[:, 2] - bboxes[:, 0]
        hs = bboxes[:, 3] - bboxes[:, 1]
        whs = torch.stack([ws, hs], dim=1)
        #vis(img, locations, locations, img_metas)
        for i in range(len(self.num_iter)):
            deformer = self.__getattr__("deformer" + str(i))
            if i == 0:
                pred_location = self.evolve(
                    deformer, features, locations, image_idx, image_sizes, whs
                )
            else:
                pred_location = self.evolve(
                    deformer,
                    features,
                    pred_location,
                    image_idx,
                    image_sizes,
                    whs,
                    att=True,
                )
            #vis(img, pred_location, pred_location, img_metas)
            location_preds.append(pred_location)

        return self._location_postprocess(location_preds, proposals, image_idx, image_sizes_ori, scale_factors)
    
    def _location_postprocess(
        self, location_preds, proposals, image_idx, image_sizes, scale_factors
    ):
        results = []
        if image_idx is None:
            pred_per_im = location_preds[-1]
            pred_per_im /= pred_per_im.new_tensor(scale_factors)
            det_polys = PolygonPoints(pred_per_im)
            cur_size = image_sizes[0]
            output_height = cur_size[0]
            output_width = cur_size[1]
            #image_size_ori : [h, w]
            det_rles = get_polygon_rles(
                det_polys.flatten(), (output_height, output_width)
            )
            return det_rles

        # per im
        #change it into coco evaluator format
        for i, _ in enumerate(proposals):
            pred_per_im = location_preds[-1][image_idx == i]  # N x 128 x 2
            det_polys = PolygonPoints(pred_per_im)
            #try image size
            cur_size = image_sizes[image_idx == i]
            output_height = cur_size[0]
            output_width = cur_size[1]
            det_rles = get_polygon_rles(
                det_polys.flatten(), (output_height, output_width)
            )
            results.append(det_rles)
        return results

    def forward_train(self,
                      x,
                      gt_bboxes,
                      gt_masks,
                      gt_semantic_seg,
                      img_metas,
                      gt_bboxes_ignore=None,
                      positive_infos=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            positive_infos (list[:obj:`InstanceData`], optional): Information
                of positive samples. Used when the label assignment is
                done outside the MaskHead, e.g., in BboxHead in
                YOLACT or CondInst, etc. When the label assignment is done in
                MaskHead, it would be None, like SOLO. All values
                in it should have shape (num_positive_samples, *).
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if positive_infos is None:
            seg_preds, outs = self(x)
        else:
            outs = self(x, positive_infos)

        loss = self.loss(
            outs,
            seg_preds,
            gt_bboxes=gt_bboxes,
            gt_masks=gt_masks,
            gt_semantic_seg=gt_semantic_seg,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            positive_infos=positive_infos,
            **kwargs)
        return loss

    def simple_test(self,
                    feats,
                    proposals,
                    img_metas,
                    rescale=False,
                    instances_list=None,
                    **kwargs):
        """Test function without test-time augmentation.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            instances_list (list[obj:`InstanceData`], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.
        Returns:
            list[obj:`InstanceData`]: Instance segmentation \
                results of each image after the post process. \
                Each item usually contains following keys. \
                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        if instances_list is None:
            _, outs = self(feats)
        else:
            outs = self(feats, instances_list=instances_list)
        results_list = self.get_results(
            outs,
            proposals,
            img_metas,
            **kwargs)
        return results_list

"""def vis(image, poly_sample_locations, poly_sample_targets):
    import matplotlib.pyplot as plt

    w = image.shape[2]
    h = image.shape[3]
    image_vis = image.reshape(-1, w, h)
    image_vis = image_vis.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
    #image_vis = image_vis.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    poly_sample_locations = poly_sample_locations.cpu().numpy()
    poly_sample_targets = poly_sample_targets.cpu().numpy()
    colors = (
        np.array([[1, 1, 198], [51, 1, 148], [101, 1, 98], [151, 1, 48], [201, 1, 8]])
        / 255.0
    )

    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()

    ax.imshow(image_vis)

    for i, (loc, target) in enumerate(zip(poly_sample_locations, poly_sample_targets)):
        offsets = target - loc
        for j in range(len(loc)):
            if j == 0:
                ax.text(loc[:1, 0], loc[:1, 1], str(i))
            ax.arrow(loc[j, 0], loc[j, 1], offsets[j, 0], offsets[j, 1])

        ax.plot(loc[0:, 0], loc[0:, 1], color="g", marker="1")
        ax.plot(target[0:, 0], target[0:, 1], marker="1", color=colors[i % 5].tolist())

    plt.show()
    fig.savefig("/home/sjtu/scratch/shaoyinkang/dance_mmdetection/tmp.jpg", bbox_inches="tight", pad_inches=0)"""