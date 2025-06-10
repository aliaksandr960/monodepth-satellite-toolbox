import rasterio
import cv2
import numpy as np
import glob
import os
from tqdm import trange, tqdm
from rasterio.windows import Window
import json
import copy
import math
import statistics


def center_weight_mask(shape, power=1.0):
    h, w = shape
    y, x = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(x**2 + y**2)  # Distance from center in normalized space
    weight = 1.0 - dist  # Closer to center = higher value
    weight = np.clip(weight, 0, 1)  # Remove negatives
    weight **= power  # Sharpen the center if needed
    return weight.astype(np.float32) + 1e-6


def make_patch_map(patch_dir, patch_ext):
    patch_path_list = sorted(glob.glob(f'{str(patch_dir)}/**.{str(patch_ext)}'))

    patch_path_map = dict()
    for p in patch_path_list:
        p_name = os.path.basename(p)[:-len(patch_ext)-1]
        p_name_splitted = p_name.split('_')
        r, c = int(p_name_splitted[0]), int(p_name_splitted[1])
        if r not in patch_path_map:
            patch_path_map[r] = dict()
        patch_path_map[r][c] = p
    return patch_path_map


def merge_and_save_patch_grid(src_raster_path, dst_raster_path, patch_path_map):

    # Get src file dimentions and profile
    profile = None
    with rasterio.open(src_raster_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width


    patch_shape = cv2.imread(patch_path_map[0][0], cv2.IMREAD_UNCHANGED).shape
    patch_size = patch_shape[0]
    patch_half_size = patch_size // 2

    image_h_offset = (height % patch_half_size) // 2
    image_w_offset = (width % patch_half_size) // 2

    n_rows = len([i for i in patch_path_map.keys()])
    n_cols = len([i for i in patch_path_map[0].keys()])

    out = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    patch_shape = cv2.imread(patch_path_map[0][0], cv2.IMREAD_UNCHANGED).shape
    weights = center_weight_mask(patch_shape)

    for nh in trange(n_rows):
        for nw in range(n_cols):
            patch = cv2.imread(patch_path_map[nh][nw], cv2.IMREAD_UNCHANGED)
            top = (nh * patch_half_size) + image_h_offset
            left = (nw * patch_half_size) + image_w_offset
            h, w = patch.shape
            out[top:top+h, left:left+w] += patch * weights
            weight_sum[top:top+h, left:left+w] += weights

    # Avoid division by zero
    data_array = np.divide(out, weight_sum, out=np.zeros_like(out), where=weight_sum!=0)

    # Update profile to singleband FP32
    profile.update(
        dtype=rasterio.float32,
        count=1,
    )

    with rasterio.open(dst_raster_path, 'w', **profile) as dst:
        for nh in trange(n_rows):
            for nw in range(n_cols):
                
                h1 = (nh * patch_half_size) + image_h_offset
                w1 = (nw * patch_half_size) + image_w_offset
                h2 = h1 + patch_size
                w2 = w1 + patch_size

                data = data_array[h1:h2, w1:w2]
                
                window = Window(w1, h1, patch_size, patch_size)
                dst.write(data, 1, window=window)


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    
    os.makedirs(config['analytics_dir'], exist_ok=False)

    src_raster_path = config['src_raster_path']

    merge_list = [
        (config['normalized_heightmap_dir'], config['normalized_heightmap_ext'], 'normalized_heightmap.tif'),
        (config['wall_dir'], config['wall_ext'], 'walls.tif'),
        (config['fall_dir'], config['fall_ext'], 'falls.tif'),
    ]


    for patch_dir, patch_ext, dst_name in tqdm(merge_list):
        patch_path_map = make_patch_map(patch_dir, patch_ext)
        dst_path = os.path.join(config['analytics_dir'], dst_name)
        merge_and_save_patch_grid(src_raster_path, dst_path, patch_path_map)


    directions_path_list = sorted(glob.glob(f'{str(config["direction_dir"])}/**.json'))

    rkh_list = []
    rkw_list = []
    for d in directions_path_list:
        with open(d, 'r') as file:
            directions_data = json.load(file)
        kh, kw = directions_data['kh'], directions_data['kw']

        dk = math.dist((0, 0), (kh, kw))
        rkh_list.append(kh / dk)
        rkw_list.append(kw / dk)

    mkh = statistics.median(rkh_list)
    mkw = statistics.median(rkw_list)
    std_kh = statistics.stdev(rkh_list)
    std_kw = statistics.stdev(rkw_list)

    with open(os.path.join(config['analytics_dir'], 'directions.json'), 'w') as f:
        json.dump({'mkh': float(mkh), 'mkw': float(mkw), 'std_kh': float(std_kh), 'std_kw': float(std_kw)}, f)

    return True