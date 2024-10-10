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
      setattr(obs[i], "%s_depth"%camera,
              image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (camera, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale))
      near = obs[i].misc['%s_camera_near'%(camera)]
      far = obs[i].misc['%s_camera_far'%(camera)]
      setattr(obs[i], "%s_depth"%camera,
              near +  getattr(obs[i], "%s_depth"%camera) * (far - near))
      
      # POINT_CLOUD
      setattr(obs[i], "%s_point_cloud"%camera,
              VisionSensor.pointcloud_from_depth_and_camera_params(
                getattr(obs[i],"%s_depth"%camera),
                obs[i].misc['%s_camera_extrinsics'%camera],
                obs[i].misc['%s_camera_intrinsics'%camera]))

    # # RGB
    # obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
    # obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
    # obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
    # obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

    # # DEPTH
    # obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
    # obs[i].front_depth = near + obs[i].front_depth * (far - near)

    # obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
    # obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)

    # obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
    # obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)

    # obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), depth_scale)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
    # obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

    # # POINT_CLOUD
    # obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth, 
    #                                                                                 obs[i].misc['front_camera_extrinsics'],
    #                                                                                 obs[i].misc['front_camera_intrinsics'])
    # obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].left_shoulder_depth, 
    #                                                                                         obs[i].misc['left_shoulder_camera_extrinsics'],
    #                                                                                         obs[i].misc['left_shoulder_camera_intrinsics'])
    # obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].right_shoulder_depth, 
    #                                                                                          obs[i].misc['right_shoulder_camera_extrinsics'],
    #                                                                                          obs[i].misc['right_shoulder_camera_intrinsics'])
    # obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth, 
    #                                                                                        obs[i].misc['wrist_camera_extrinsics'],
    #                                                                                        obs[i].misc['wrist_camera_intrinsics'])
    
  return obs