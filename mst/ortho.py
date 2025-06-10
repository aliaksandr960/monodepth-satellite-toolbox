import numpy as np
import os
import copy
import json
import cv2
import rasterio
from scipy import signal



def make_transform_f(kx, ky, bx=0, by=0):
    def f(i):
        return (i[0] + kx*i[2] + bx, i[1] + ky*i[2] + by, i[2]) 
    return f


def ortho_from_pointcloud(points, height, width, res=1, colors=None,
                          falls=None, fall_threshold=0.95):
    
    x, y, z = points[:,0], points[:,1], points[:,2]
    x_min, y_min = x.min(), y.min()
    
    i = ((x - x_min) / res).astype(int)
    j = ((y - y_min) / res).astype(int)
    
    img = np.zeros((height, width), dtype=np.float32)
    if colors is not None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
    
    depth = np.full((height, width), -np.inf)
    for idx in range(len(points)):
        jj, ii = i[idx], j[idx]
        if jj >= height:
            continue
        if ii >= width:
            continue

        if z[idx] > depth[jj, ii]:
            if falls is not None:
                fall_value = falls[idx]
                if fall_value > fall_threshold:
                    continue
                
            depth[jj, ii] = z[idx]
            
            if colors is not None:
                img[jj, ii] = colors[idx]
            else:
                img[jj, ii] = z[idx]
    
    return img


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    os.makedirs(config['ortho_dir'], exist_ok=False)

    
    color_path = os.path.join(config['src_raster_path'])
    hmap_path = os.path.join(config['analytics_dir'], 'normalized_heightmap.tif')
    fall_path = os.path.join(config['analytics_dir'], 'falls.tif')
    direction_path = os.path.join(config['analytics_dir'], 'directions.json')

    with open(direction_path, 'r') as file:
        directions_data = json.load(file)
    kh, kw = directions_data['mkh'], directions_data['mkw']

    color = cv2.imread(color_path)
    hmap = cv2.imread(hmap_path, cv2.IMREAD_UNCHANGED)
    fall = cv2.imread(fall_path, cv2.IMREAD_UNCHANGED)

    max_h, max_w = hmap.shape
    point_list = []
    color_list = []
    fall_list = []
    for h in range(max_h):
        for w in range(max_w):
            z = hmap[h, w] 
            c = color[h, w, :]
            f = fall[h, w]
            
            point_list.append((h, w, z))
            color_list.append(c)
            fall_list.append(f)

    point_array, color_array, fall_array = np.array(point_list), np.array(color_list), np.array(fall_list)
    transform_f = make_transform_f(kh, -kw)
    transformed_point_array = np.apply_along_axis(transform_f, axis=-1, arr=point_array)

    ortho_z = ortho_from_pointcloud(transformed_point_array, max_h, max_w, res=1,
                                    colors=None, falls=fall_array, fall_threshold=0.85)
    
    ortho_color = ortho_from_pointcloud(transformed_point_array, max_h, max_w, res=1,
                                        colors=color_array, falls=fall_array, fall_threshold=0.85)
    
    mask = ortho_z > 0
    kernel = np.ones([3, 3])
    mask = signal.convolve2d(mask, kernel, boundary='symm', mode='same')
    mask = mask > 5

    ortho_z = ortho_z * mask
    ortho_color = ortho_color * np.stack([mask, mask, mask], axis=-1)

    with rasterio.open(color_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.uint8,
        count=3,
    )

    # Write to file
    ortho_color_path = os.path.join(config['ortho_dir'], 'color.tif')
    with rasterio.open(ortho_color_path, 'w', **profile) as dst:
        dst.write(ortho_color[...,[2,1,0]].transpose(2,0,1))  # shape must be (bands, rows, cols)


    profile.update(
        dtype=rasterio.float32,
        count=1,
    )

    ortho_height_path = os.path.join(config['ortho_dir'], 'height.tif')
    with rasterio.open(ortho_height_path, 'w', **profile) as dst:
        dst.write(np.expand_dims(ortho_z, axis=0))
    
    return True