import numpy as np
import torch

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