from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import json
import copy
import glob
import os
from tqdm import tqdm


# Remove background by substracting min surface
DEFAULT_MIN_POOL_KERNEL = 127
DEFAULT_MIN_POOL_PADDING = 63
DEFAULT_GAUSSIAN_BLUR_SIGMA = 63


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


def remove_background(depth, min_pool_kernel=DEFAULT_MIN_POOL_KERNEL,
                      min_pool_padding=DEFAULT_MIN_POOL_PADDING, gaussian_blur_sigma=DEFAULT_GAUSSIAN_BLUR_SIGMA):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    reversed_depth_minpool = minpool2d_nearest_pad(depth, kernel_size=min_pool_kernel, padding=min_pool_padding)
    reversed_depth_minpool_blured = gaussian_filter(reversed_depth_minpool, sigma=gaussian_blur_sigma)
    depth_nobackground = depth - reversed_depth_minpool_blured
    depth_nobackground = (depth_nobackground - np.min(depth_nobackground)) / (np.max(depth_nobackground) - np.min(depth_nobackground))
    return depth_nobackground


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    
    os.makedirs(config['heightmap_dir'], exist_ok=False)

    depthmap_path_list = glob.glob(f'{str(config["depthmap_dir"])}/**.{str(config["depthmap_ext"])}')

    for p in tqdm(depthmap_path_list):

        depthmap_name = os.path.basename(p)[:-len(config['depthmap_ext'])-1]
        heightmap_name = f'{depthmap_name}.{str(config["heightmap_ext"])}'
        heightmap_path = os.path.join(config['heightmap_dir'], heightmap_name)
        
        depth = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        heightmap = remove_background(1 - depth,
                                     config['heightmap_min_pool_kernel_1'],
                                     config['heightmap_min_pool_padding_1'],
                                     config['heightmap_gaussian_blur_sigma_1'])
        
        heightmap = remove_background(heightmap,
                                     config['heightmap_min_pool_kernel_2'],
                                     config['heightmap_min_pool_padding_2'],
                                     config['heightmap_gaussian_blur_sigma_2'])


        cv2.imwrite(heightmap_path, heightmap)

    return True