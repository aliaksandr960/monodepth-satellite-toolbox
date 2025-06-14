from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import json
import copy
import os
import glob
import cv2
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_depthpro_func():
    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    
    def f(image):
        image = Image.fromarray(image)
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        post_processed_output = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)],)
        depth_numpy = post_processed_output[0]["predicted_depth"].cpu().numpy()
        
        return depth_numpy

    return f


def make_depthanything_func():
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    def f(image):
        image = Image.fromarray(image)
        inputs = image_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        post_processed_output = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)],)
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth_numpy = predicted_depth.detach().cpu().numpy()
        
        return np.max(depth_numpy) - depth_numpy
        
    return f


def using_config(config):
    if type(config) is str:
        with open(config, 'r') as file:
            config = json.load(file)

    config = copy.deepcopy(config)
    
    os.makedirs(config['depthmap_dir'], exist_ok=False)

    depth_processor_name = config['depth_processor']
    if depth_processor_name == 'depthanything':
        processor = make_depthanything_func()
    elif depth_processor_name == 'depthpro':
        processor = make_depthpro_func()
    else:
        raise ValueError('Invalid processor name')

    patch_path_list = glob.glob(f'{str(config["patch_dir"])}/**.{str(config["patch_ext"])}')

    for p in tqdm(patch_path_list):

        patch_name = os.path.basename(p)[:-len(config['patch_ext'])-1]
        depthmap_name = f'{patch_name}.{str(config["depthmap_ext"])}'
        depthmap_path = os.path.join(config['depthmap_dir'], depthmap_name)
        
        image = cv2.imread(p)
        depth = processor(image)
        cv2.imwrite(depthmap_path, depth)

    return True

