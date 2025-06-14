import math
import json
import cv2
import os
import copy
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_kernel = torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, 0, 0]]]]).float().to(device)
top_kernel_2c = torch.cat([top_kernel, top_kernel], dim=0).to(device)

left_kernel = torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]]).float().to(device)
left_kernel_2c = torch.cat([left_kernel, left_kernel], dim=0).to(device)

center_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).float().to(device)
center_kernel_2c = torch.cat([center_kernel, center_kernel], dim=0).to(device)

right_kernel = torch.tensor([[[[0, 0, 0], [0, 0, 1], [0, 0, 0]]]]).float().to(device)
right_kernel_2c = torch.cat([right_kernel, right_kernel], dim=0).to(device)

bottom_kernel = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 1, 0]]]]).float().to(device)
bottom_kernel_2c = torch.cat([bottom_kernel, bottom_kernel], dim=0).to(device)

kernels_1d_list = [top_kernel,    left_kernel,    center_kernel,    right_kernel,    bottom_kernel]
kernels_2d_list = [top_kernel_2c, left_kernel_2c, center_kernel_2c, right_kernel_2c, bottom_kernel_2c]


def extract_z_from_1d(x):
    results = []
    for k in kernels_1d_list:
        shifted = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=1, dilation=1, groups=1)
        results.append(shifted[:, 0, :, :])
    return results


def angle_from_unit_vector(vec):
    angle_rad = torch.atan2(vec[:, 1], vec[:, 0])  # atan2(y, x)
    angle_deg = angle_rad * 180.0 / torch.pi
    return angle_deg


def rotate_tensor(img, angle):

    b, _, h, w = img.size()
    theta = torch.zeros(b, 2, 3, device=img.device, dtype=img.dtype)

    angle = angle * torch.pi / 180.0  # convert to radians
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos

    tx = (1 - cos) * 1 - sin * 1
    ty = sin * 1 + (1 - cos) * 1

    theta[:, 0, 2] = tx / (w / 2.0)
    theta[:, 1, 2] = ty / (h / 2.0)

    grid = nn.functional.affine_grid(theta, img.size(), align_corners=False)
    return nn.functional.grid_sample(img, grid, align_corners=False)


def normalize_heightmap(heightmap, kh, kw):
    dk = math.dist((0, 0), (kh, kw))
    heightmap_n = heightmap * dk
    return heightmap_n


def extract_walls_and_falls(heightmap, kh, kw,
                            wall_shade_min_coeff=0.25, fall_shade_min_coeff=0.1,
                            wall_blur_kernel_size=3, wall_rotataion_pad=256):

    init_d = 1 / math.dist((0,0), (kh, kw))

    # Rotate depth and shades
    khkw = torch.tensor([[kw, kh], ])
    rotation_a = angle_from_unit_vector(khkw) - 90

    depth_t = torch.from_numpy(heightmap).float()
    depth_t = torch.unsqueeze(torch.unsqueeze(depth_t, dim=0), dim=0)
    
    depth_t_padded = nn.functional.pad(depth_t, (wall_rotataion_pad, wall_rotataion_pad, wall_rotataion_pad, wall_rotataion_pad))
    depth_t_rotated = rotate_tensor(depth_t_padded, rotation_a)

    z = depth_t_rotated
    tz, _, cz, _, bz = extract_z_from_1d(torchvision.transforms.functional.gaussian_blur(z, kernel_size=wall_blur_kernel_size))
    
    # Wall shades
    wt_shade = (cz - tz) / (init_d - init_d * wall_shade_min_coeff)
    wt_shade_min = torch.clip(wt_shade, 0, 1)
    
    wb_shade = (bz - cz) / (init_d - init_d * wall_shade_min_coeff)
    wb_shade_min = torch.clip(wb_shade, 0, 1)
    
    w_shade_map, _ = torch.max(torch.stack([wt_shade_min, wb_shade_min], dim=0), dim=0)
    
    shade_map_unrotated = rotate_tensor(torch.unsqueeze(w_shade_map.float(), dim=0), -rotation_a)
    walls = shade_map_unrotated[:, :, wall_rotataion_pad:-wall_rotataion_pad, wall_rotataion_pad:-wall_rotataion_pad]
    
    
    # Falls shades
    ft_shade = cz - tz
    ft_shade_min = ft_shade < -(init_d + init_d * fall_shade_min_coeff)
    fb_shade = bz - cz
    fb_shade_min = fb_shade < -(init_d + init_d * fall_shade_min_coeff)
    f_shade_map = (ft_shade_min + fb_shade_min) > 0
    f_shade_map_unrotated = rotate_tensor(torch.unsqueeze(f_shade_map.float(), dim=0), -rotation_a)
    falls = f_shade_map_unrotated[:, :, wall_rotataion_pad:-wall_rotataion_pad, wall_rotataion_pad:-wall_rotataion_pad]

    return walls.float()[0, 0, :, :].detach().cpu().numpy(), falls.float()[0, 0, :, :].detach().cpu().numpy()


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    
    os.makedirs(config['normalized_heightmap_dir'], exist_ok=False)
    os.makedirs(config['wall_dir'], exist_ok=False)
    os.makedirs(config['fall_dir'], exist_ok=False)


    heightmap_path_list = sorted(glob.glob(f'{str(config["heightmap_dir"])}/**.{str(config["heightmap_ext"])}'))
    directions_path_list = sorted(glob.glob(f'{str(config["direction_dir"])}/**.json'))

    for h, d in zip(heightmap_path_list, directions_path_list):
        heightmap_name = os.path.basename(h)[:-len(config['heightmap_ext'])-1]
        direction_name = os.path.basename(d)[:-len('json')-1]
        if heightmap_name != direction_name:
            raise ValueError('Heightmap and direction files not match')


    for ph, pd in tqdm(zip(heightmap_path_list, directions_path_list)):
        heightmap_name = os.path.basename(ph)[:-len(config['heightmap_ext'])-1]
        heightmap = cv2.imread(ph, cv2.IMREAD_UNCHANGED)
        
        with open(pd, 'r') as file:
            directions_data = json.load(file)
        kh, kw = directions_data['kh'], directions_data['kw']

        normalized_heightmap = normalize_heightmap(heightmap, kh, kw)

        normalized_heightmap_name = f'{heightmap_name}.{str(config["normalized_heightmap_ext"])}'
        normalized_heightmap_path = os.path.join(config['normalized_heightmap_dir'], normalized_heightmap_name)
        cv2.imwrite(normalized_heightmap_path, normalized_heightmap)

        walls, falls = extract_walls_and_falls(heightmap, kh, kw,
                                               wall_shade_min_coeff=config["wall_shade_min_coeff"],
                                               fall_shade_min_coeff=config["fall_shade_min_coeff"],
                                               wall_blur_kernel_size=config["wall_blur_kernel_size"],
                                               wall_rotataion_pad=config["wall_rotataion_pad"])
        
        wall_name = f'{heightmap_name}.{str(config["wall_ext"])}'
        wall_path = os.path.join(config['wall_dir'], wall_name)
        cv2.imwrite(wall_path, walls)

        fall_name = f'{heightmap_name}.{str(config["fall_ext"])}'
        fall_path = os.path.join(config['fall_dir'], fall_name)
        cv2.imwrite(fall_path, falls)

    return True