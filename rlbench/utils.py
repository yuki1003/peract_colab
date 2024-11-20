# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/utils.py

import os
import pickle
import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT  = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1

# functions
def get_stored_demo(data_path, index, cameras = CAMERAS, depth_scale = DEPTH_SCALE):
  episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
  
  # low dim pickle file
  with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
    obs = pickle.load(f)

  # variation number
  with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
    obs.variation_number = pickle.load(f)

  num_steps = len(obs)
  for i in range(num_steps):

    for camera in cameras:
      
      # RGB
      setattr(obs[i], "%s_rgb"%camera,
              np.array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_RGB), IMAGE_FORMAT % i))))
      
      # DEPTH
      # depth_frame = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), 1)
      # print(f"Depth Image: {depth_frame.min()}, {depth_frame.max()}\n{depth_frame}")
      setattr(obs[i], "%s_depth"%camera,
              image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale))
      # depth_frame = getattr(obs[i], "%s_depth"%camera)
      # print(f"Depth Before: {depth_frame.min()}, {depth_frame.max()}\n{depth_frame}")
      near = obs[i].misc['%s_camera_near'%(camera)]
      far = obs[i].misc['%s_camera_far'%(camera)]
      # print(f"depth camera: {near} - {far}")
      setattr(obs[i], "%s_depth"%camera,
              near +  getattr(obs[i], "%s_depth"%camera) * (far - near))
      # depth_frame = getattr(obs[i], "%s_depth"%camera)
      # print(f"Depth After: {depth_frame.min()}, {depth_frame.max()}\n{depth_frame}")
      
      # POINT_CLOUD
      setattr(obs[i], "%s_point_cloud"%camera,
              VisionSensor.pointcloud_from_depth_and_camera_params(
                getattr(obs[i],"%s_depth"%camera),
                obs[i].misc['%s_camera_extrinsics'%camera],
                obs[i].misc['%s_camera_intrinsics'%camera]))

  return obs