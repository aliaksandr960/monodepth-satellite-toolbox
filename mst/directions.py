import torch
import torch.nn as nn
from skimage.transform import rescale
import numpy as np
import json
from skimage.morphology import skeletonize
from tqdm import trange
import copy
import os
import glob
from tqdm import tqdm
import cv2


def minpool2d_nearest_pad(i, kernel_size, stride=1, padding=0):
    if padding > 0:
        input_padded = np.pad(i, 
                              pad_width=((padding, padding), (padding, padding)),
                              mode='edge')
    else:
        input_padded = i

    H, W = input_padded.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            output[i, j] = np.min(input_padded[h_start:h_end, w_start:w_end])

    return output


DEFAULT_DIRECTION_SHADE_KERNEL_SIZE = 9
DEFAULT_DIRECTION_SHADE_THRESHOLD = 0.05
DEFAULT_DIRECTION_SHADE_COUNT = 24
DEFAULT_DIRECTION_SHADE_SKIP_FIRST = 8
DEFAULT_DIRECTION_SHADE_SKIP_LAST = 4
DEFAULT_DIRECTION_SHADE_CC_CROP = 7
DEFAULT_DIRECTION_SHADE_STEP = 2
DEFAULT_DIRECTION_SKELETON_MAXPOOL_KERNEL_SIZE = 3


def calculate_direction_coeffs(heightmap,
                               direction_shade_kernel_size=DEFAULT_DIRECTION_SHADE_KERNEL_SIZE,
                               direction_shade_threshold=DEFAULT_DIRECTION_SHADE_THRESHOLD,
                               direction_shades_count=DEFAULT_DIRECTION_SHADE_COUNT,
                               direction_shade_skip_first=DEFAULT_DIRECTION_SHADE_SKIP_FIRST,
                               direction_shades_skip_last=DEFAULT_DIRECTION_SHADE_SKIP_LAST,
                               direction_shades_cc_crop=DEFAULT_DIRECTION_SHADE_CC_CROP,
                               direction_shades_step=DEFAULT_DIRECTION_SHADE_STEP,
                               direction_skeleton_maxpool_kernel_size=DEFAULT_DIRECTION_SKELETON_MAXPOOL_KERNEL_SIZE,
                               **kwargs
                               ):
    p = heightmap - minpool2d_nearest_pad(heightmap, kernel_size=direction_shade_kernel_size, stride=1,
                                          padding=direction_shade_kernel_size//2)
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    p = p > direction_shade_threshold

    d = heightmap * p
    d = (d - np.min(d)) / (np.max(d) - np.min(d))


    shades_q = (heightmap * direction_shades_count).astype(np.uint8)
    shades_skeletons_list = []
    for i in range(direction_shades_count):
        shade_item = shades_q == i
        shades_skeleton_item = skeletonize(shade_item)
        h, w = shade_item.shape
        shades_skeleton_t = torch.from_numpy(shades_skeleton_item).reshape(1, 1, h, w).float()
        shades_skeleton_t = nn.functional.max_pool2d(shades_skeleton_t,
                                                    kernel_size=direction_skeleton_maxpool_kernel_size,
                                                    stride=1, padding=direction_skeleton_maxpool_kernel_size//2,
                                                    dilation=1, ceil_mode=False, return_indices=False)
        shades_skeletons_list.append(shades_skeleton_t)


    cross_correlation_list = []
    for n  in trange(direction_shade_skip_first, len(shades_skeletons_list) - direction_shades_skip_last):
        skeleton = shades_skeletons_list[n]
        previous_skeleton = shades_skeletons_list[n-direction_shades_step]
        cc = torch.nn.functional.conv2d(skeleton, previous_skeleton[:, :, direction_shades_cc_crop:-direction_shades_cc_crop, direction_shades_cc_crop:-direction_shades_cc_crop],
                                            bias=None, stride=1, padding=0, dilation=1, groups=1)
        cross_correlation_list.append(cc.cpu().numpy())

    cross_correlation_sum = np.zeros_like(cross_correlation_list[0])
    for cc in cross_correlation_list:
        cross_correlation_sum += cc
    cross_correlation_sum_2d = cross_correlation_sum[0,0, :, :]


    ap_rescaled = rescale(cross_correlation_sum_2d, direction_shades_count//2, anti_aliasing=False)
    ah, aw = ap_rescaled.shape
    cah, caw = ah // 2, aw // 2

    flat_index = np.argmax(ap_rescaled)
    th, tw = np.unravel_index(flat_index, ap_rescaled.shape)
    kh, kw = th - cah, tw - caw
    return kh, kw


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    
    os.makedirs(config['direction_dir'], exist_ok=False)

    heightmap_path_list = glob.glob(f'{str(config["heightmap_dir"])}/**.{str(config["heightmap_ext"])}')

    for p in tqdm(heightmap_path_list):

        heightmap_name = os.path.basename(p)[:len(config['depthmap_ext'])]
        direction_name = f'{heightmap_name}.json'
        direction_path = os.path.join(config['direction_dir'], direction_name)
        heightmap = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        kh, kw = calculate_direction_coeffs(heightmap, **config)
        with open(direction_path, 'w') as f:
            json.dump({'kh': float(kh), 'kw': float(kw)}, f)

    return True