import os
import sys
import copy
import shutil
from addict import Dict as ADict

from mst.split import using_config as split
from mst.depthmaps import using_config as depthmaps
from mst.heightmaps import using_config as heightmaps
from mst.directions import using_config as directions
from mst.basic_analytics import using_config as basic_analytics
from mst.merge_analytics import using_config as merge_analytics
from mst.ortho import using_config as ortho
from mst.point_cloud import using_config as point_cloud

def remove_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' and its contents have been removed.")
    except FileNotFoundError:
        pass


defalut_config = ADict()
defalut_config.reconstruction_path = None

# Patch generation
defalut_config.patch_dir_name = 'patches'
defalut_config.src_raster_name = 'raster.tif'



defalut_config.patch_ext = 'jpg'
defalut_config.patch_size = 768
defalut_config.band_list = [3, 2, 1]

# Depthmaps generation
defalut_config.depthmap_dir_name = 'depathmaps'


defalut_config.depth_processor = 'depthanything'
# defalut_config.depth_processor = 'depthpro' 
defalut_config.depthmap_ext = 'tif'

# Heightmap generation
defalut_config.heightmap_dir_name = 'heightmaps'


defalut_config.heightmap_ext = 'tif'

defalut_config.heightmap_min_pool_kernel_1 = 127
defalut_config.heightmap_min_pool_padding_1 = 63
defalut_config.heightmap_gaussian_blur_sigma_1 = 63

defalut_config.heightmap_min_pool_kernel_2 = 127
defalut_config.heightmap_min_pool_padding_2 = 63
defalut_config.heightmap_gaussian_blur_sigma_2 = 15.75

# Directions generation
defalut_config.direction_dir_name = 'directions'


defalut_config.direction_shade_kernel_size = 9
defalut_config.direction_shade_threshold = 0.025
defalut_config.direction_shades_count = 24
defalut_config.direction_shade_skip_first = 8
defalut_config.direction_shades_skip_last = 4
defalut_config.direction_shades_cc_crop = 11
defalut_config.direction_shades_step = 2
defalut_config.direction_skeleton_maxpool_kernel_size = 3

# Normalized height generation
defalut_config.normalized_heightmap_dir_name = 'normalized_heightmaps'

defalut_config.normalized_heightmap_ext = 'tif'

defalut_config.wall_dir_name = 'walls'

defalut_config.wall_ext = 'tif'

defalut_config.fall_dir_name = 'falls'

defalut_config.fall_ext = 'tif'

defalut_config.wall_shade_min_coeff = 0.35
defalut_config.fall_shade_min_coeff = 0.1
defalut_config.wall_blur_kernel_size = 3
defalut_config.wall_rotataion_pad = 256

# Merge analytics
defalut_config.analytics_dir_name = 'analytics'

# Make ortho
defalut_config.ortho_dir_name = 'ortho'
defalut_config.point_cloud_downscale = None
defalut_config.fall_threshold = 0.85
defalut_config.ortho_outlier_filter_treshold = 5

# Make point cloud
defalut_config.point_cloud_dir_name = 'pointcloud'


def run_reconstruction(path='./test_reconstruction', **kwargs):
    print('# Run:', str(path))
    actual_config = copy.deepcopy(defalut_config)
    actual_config.reconstruction_path = path
    actual_config.update(kwargs)

    actual_config.src_raster_path = os.path.join(actual_config.reconstruction_path, actual_config.src_raster_name)
    actual_config.patch_dir = os.path.join(actual_config.reconstruction_path, actual_config.patch_dir_name)
    actual_config.depthmap_dir = os.path.join(actual_config.reconstruction_path, actual_config.depthmap_dir_name)
    actual_config.heightmap_dir = os.path.join(actual_config.reconstruction_path, actual_config.heightmap_dir_name)
    actual_config.direction_dir = os.path.join(actual_config.reconstruction_path, actual_config.direction_dir_name)
    actual_config.normalized_heightmap_dir = os.path.join(actual_config.reconstruction_path, actual_config.normalized_heightmap_dir_name)
    actual_config.wall_dir = os.path.join(actual_config.reconstruction_path, actual_config.wall_dir_name)
    actual_config.fall_dir = os.path.join(actual_config.reconstruction_path, actual_config.fall_dir_name)
    actual_config.analytics_dir = os.path.join(actual_config.reconstruction_path, actual_config.analytics_dir_name)
    actual_config.ortho_dir = os.path.join(actual_config.reconstruction_path, actual_config.ortho_dir_name)
    actual_config.point_cloud_dir = os.path.join(actual_config.reconstruction_path, actual_config.point_cloud_dir_name)

    # Pipeline run
    print('Split started.')
    split(actual_config)
    print('Split done!')

    print('Depthmaps started.')
    depthmaps(actual_config)
    print('Depthmaps done!')

    print('Heightmaps started.')
    heightmaps(actual_config)
    print('Heightmaps done!')

    print('Directions started.')
    directions(actual_config)
    print('Directions done!')

    print('Basic_analytics started.')
    basic_analytics(actual_config)
    print('Basic_analytics done!')

    print('Merge_analytics started.')
    merge_analytics(actual_config)
    print('Merge_analytics done!')

    print('Ortho started.')
    ortho(actual_config)
    print('Ortho done!')

    print('Point_cloud started.')
    point_cloud(actual_config)
    print('Point_cloud done!')


if __name__ == "__main__":
    run_reconstruction(path=sys.argv[1])