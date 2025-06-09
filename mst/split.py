import cv2
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import json
import copy


def split_raster(src_raster_path, patch_size, band_list, patch_dir, patch_name_func, **kwargs):
    if patch_size % 2 > 0:
        raise ValueError('Patch size should be odd number')
    patch_half_size = patch_size // 2


    with rasterio.open(src_raster_path) as src:
        height = src.height
        width = src.width 
        
        image_h_offset = (height % patch_half_size) // 2
        image_w_offset = (width % patch_half_size) // 2

        h1 = image_h_offset
        nh = 0
        while((h1 + patch_size) <= height):
            
            nw = 0
            w1 = image_w_offset
            while((w1 + patch_size) <= width):
                h1 = (nh * patch_half_size) + image_h_offset
                w1 = (nw * patch_half_size) + image_w_offset
                window = Window(w1, h1, patch_size, patch_size)
                
                band_stack_list = []
                for nb in band_list:
                    band_stack_list.append(src.read(nb, window=window))
                patch = np.stack(band_stack_list, axis=-1)
                patch_path = os.path.join(patch_dir, patch_name_func(nh, nw))
                cv2.imwrite(patch_path, patch)
                nw += 1
                w1 += patch_half_size
        
            h1 += patch_half_size
            nh += 1


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)

    config['patch_name_func'] = lambda r, c: f'{str(r)}_{str(c)}.{str(config["patch_ext"])}'
    os.makedirs(config['patch_dir'], exist_ok=False)

    split_raster(**config)
    return True