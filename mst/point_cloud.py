import os
import json
import copy
import numpy as np
import open3d as o3d


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)

    if os.path.exists(config['point_cloud_dir']):
        print('Skipped. Folder with point cloud found, remove it to re-do.')
        return True
    
    os.makedirs(config['point_cloud_dir'], exist_ok=False)
    point_cloud_path = os.path.join(config['point_cloud_dir'], 'point_cloud.ply')

    points = np.load(os.path.join(config['ortho_dir'], 'transformed_point_array.npy'))
    colors = np.load(os.path.join(config['ortho_dir'],'color_array.npy')) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(point_cloud_path, pcd)
    
    return True