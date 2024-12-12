# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py

import os
import pickle
from typing import List

import numpy as np
import torch
import clip

import arm.utils as utils
from arm.demo import _keypoint_discovery, _target_object_discovery, _keypoint_discovery_available

from rlbench.backend.utils import extract_obs
from rlbench.backend.observation import Observation
from rlbench.utils import get_stored_demo
from rlbench.demo import Demo

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer


VARIATION_DESCRIPTIONS_PKL = 'variation_descriptions.pkl' # the pkl file that contains language goals for each demonstration


def create_replay(batch_size: int,
                  timesteps: int,
                  save_dir: str,
                  cameras: list,
                  voxel_sizes,
                  image_size: int,
                  low_dim_size: int,
                  replay_size=3e5,):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (low_dim_size,), np.float32)) # 4 dimensions - proprioception: {gripper_open, left_finger_joint, right_finger_joint, timestep}

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, image_size, image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, image_size, image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, image_size, image_size,), np.float32)) # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),
        ReplayElement('lang_goal_embs', (max_token_seq_len, lang_emb_dim,), # extracted from CLIP's language encoder
                      np.float32),
        ReplayElement('lang_goal', (1,), object), # language goal string for debugging and visualization
        ReplayElement('gripper_state', (gripper_pose_size+1,), # NOTE: Added gripper state of the current state [*gripper_pose, gripper_open/close]
                      np.float32),
        ReplayElement('object_state', (gripper_pose_size,), # NOTE: Added target object state of the goal state same shape as gripper_state
                      np.float32)
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = UniformReplayBuffer( # all tuples in the buffer have equal sample weighting
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,), # 3 translation + 4 rotation quaternion + 1 gripper open
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []  # encoded voxel index, decoded coordinates from voxel
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

# add individual data points to replay
def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        initial_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        target_object,
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        episode_index: int,
        frame_idx: int,
        description: str = '',
        clip_model = None,
        device = 'cpu'):
    prev_action = None
    obs = initial_obs # Observation of certain demo at certain (key)frame
    for k, keypoint in enumerate(episode_keypoints):
        #TODO: WTF is this mess?
        obs_tp1 = demo[keypoint] # Keypoints of demo
        obs_tm1 = demo[max(0, keypoint - 1)] # Observation of demo before keypoint FIXME: I think this should be the previous keyframe
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes,
            rotation_resolution, crop_augmentation) # Path from previous to current keypoint?
        terminal = (k == len(episode_keypoints) - 1) #Terminal == True, if reached last episode_keypoint
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(obs, cameras, t=k, prev_action=prev_action) # Get observations for current/previous-keyframe
        
        # Language embeddings
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action) # action: gripper pose + open)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'lang_goal': np.array([f"{description}-episode_{episode_index}-frame_{frame_idx}-kp_{keypoint}"], dtype=object),
            'gripper_state': np.concatenate((obs.gripper_pose, # NOTE: Somehow this is not the initial obs? weird way it is stored in replaybuffer
                                             [float(initial_obs.gripper_open)])),
            'object_state': target_object,
        }
        
        others.update(final_obs) # Adds trans/rot-action indices, gripper_pose and lang_goal NOTE: Goals to next-keyframe
        others.update(obs_dict) # Add RGB-D, PC, collision, low_dim_state,camera ex-/intrinsics, lang_goal_embs NOTE: Current or previous-keyframe

        timeout = False
        replay.add(action, reward, terminal, timeout, **others) # Add (action, reward, terminal, timeout) and **kwargs
        
        obs = obs_tp1 # Update obs_dict

    # final step NOTE: TERMINAL/FINAL PHASE
    obs_dict_tp1 = extract_obs(obs_tp1, cameras, t=k + 1, prev_action=prev_action)
    obs_dict_tp1['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()
    
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs) # Adds trans/rot-action indices, gripper_pose ad lang_goal. NOTE: Same as last-keyframe

    replay.add_final(**obs_dict_tp1)

def fill_replay(data_path: str,
                episode_folder: str,
                replay: ReplayBuffer,
                # start_idx: int,
                # num_demos: int,
                d_indexes: List[int],
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                rotation_resolution: int,
                crop_augmentation: bool,
                depth_scale,
                use_approach: bool,
                approach_distance: float,
                stopping_delta,
                target_obj_keypoint: bool = False,
                target_obj_use_last_kp: bool = False,
                target_obj_is_avail: bool = False,
                clip_model = None,
                device = 'cpu'):
    print('Filling replay ...')
    for d_idx in d_indexes: #range(start_idx, start_idx+num_demos): # Loops through expert demos
        print("Filling demo %d" % d_idx)
        demo = get_stored_demo(data_path=data_path,
                                index=d_idx,
                               cameras=cameras,
                               depth_scale=depth_scale) # Single episode demo

        # get language goal from disk
        varation_descs_pkl_file = os.path.join(data_path, episode_folder % d_idx, VARIATION_DESCRIPTIONS_PKL)
        with open(varation_descs_pkl_file, 'rb') as f:
            descs = pickle.load(f)

        # extract keypoints
        # episode_keypoints = _keypoint_discovery(demo, stopping_delta) # Discover keypoints for current demo index
        episode_keypoints = _keypoint_discovery_available(demo, approach_distance)
        episode_keypoints = episode_keypoints if use_approach else [episode_keypoints[-1]]

        # extract (potential) target object locations NOTE: assumed - closed gripper is object location
        episode_target_object = _target_object_discovery(demo, keypoints=target_obj_keypoint, stopping_delta=stopping_delta, last_kp=target_obj_use_last_kp, is_available=target_obj_is_avail)
        
        for i, (obs, obs_episode_target_object) in enumerate(zip(demo,episode_target_object)): # Loop through frames of demo
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0: # choose only every n-th frame
                continue
            
            # obs = demo[i] # Get the observation at i-th frame
            desc = descs[0]
            target_object = episode_target_object[i]

            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]: # Key-point discovered and frame beyond 1st-keypoint
                episode_keypoints = episode_keypoints[1:] # Remove keypoint
            if len(episode_keypoints) == 0: # No episode key-points discovered
                break
            print(i, episode_keypoints)
            _add_keypoints_to_replay(
                replay, # Where to store
                obs, # Current observation (i.e. data-X)
                demo, episode_keypoints, # (i.e. data-Y)
                target_object,
                cameras, rlbench_scene_bounds, voxel_sizes, rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device,
                episode_index=d_idx, frame_idx=i) # New items)
            
    print('Replay filled with demos.')

# add individual data points to replay
def _add_keypoint_to_replay_uniform(
        replay: ReplayBuffer,
        initial_obs: Observation,
        demo: Demo,
        episode_keypoint: int, # Only 1-keypoint
        target_object, # New
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        episode_index: int,
        kf_index: int,
        is_terminal: bool,
        frame_idx: int,
        description: str = '',
        clip_model = None,
        device = 'cpu'):
    prev_action = None
    obs = initial_obs
    
    obs_tp1 = demo[episode_keypoint]
    obs_tm1 = demo[max(0, episode_keypoint - 1)]
    trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
        obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes,
        rotation_resolution, crop_augmentation)
    
    reward = float(is_terminal) * 1.0 if is_terminal else 0

    obs_dict = extract_obs(obs, cameras, t=kf_index, prev_action=prev_action) # Is time an important value? What is it used for?

    # Language embeddings
    tokens = clip.tokenize([description]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device)
    lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
    obs_dict['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

    prev_action = np.copy(action) # action: gripper pose + open

    others = {'demo': True,}
    final_obs = {
        'trans_action_indicies': trans_indicies,
        'rot_grip_action_indicies': rot_grip_indicies,
        'gripper_pose': obs_tp1.gripper_pose,
        'lang_goal': np.array([f"{description}-episode_{episode_index}-frame_{frame_idx}-kp_{episode_keypoint}"], dtype=object),
        'gripper_state': np.concatenate((obs.gripper_pose, # NOTE: Somehow this is not the initial obs? weird way it is stored in replaybuffer
                                         [float(initial_obs.gripper_open)])),
        'object_state': target_object,
    }

    others.update(final_obs)
    others.update(obs_dict)

    timeout = False
    replay.add(action, reward, is_terminal, timeout, **others)

    obs = obs_tp1

    # final step
    if is_terminal:
        obs_dict_tp1 = extract_obs(obs_tp1, cameras, t=kf_index + 1, prev_action=prev_action)
        obs_dict_tp1['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

        obs_dict_tp1.pop('wrist_world_to_cam', None)
        obs_dict_tp1.update(final_obs)

        replay.add_final(**obs_dict_tp1)

def uniform_fill_replay(data_path: str,
                        episode_folder: str,
                        replay: ReplayBuffer,
                        # start_idx: int,
                        # num_demos: int,
                        d_indexes: List[int],
                        demo_augmentation: bool,
                        demo_augmentation_every_n: int,
                        cameras: List[str],
                        rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                        voxel_sizes: List[int],
                        rotation_resolution: int,
                        crop_augmentation: bool,
                        depth_scale,
                        use_approach: bool,
                        approach_distance: float,
                        stopping_delta: float,
                        target_obj_keypoint: bool = False,
                        target_obj_use_last_kp: bool = False,
                        target_obj_is_avail: bool = False,
                        clip_model = None,
                        device = 'cpu'):
    print('Filling replay unformly ...')
    for d_idx in d_indexes: #range(start_idx, start_idx+num_demos):
        print("Filling demo %d" % d_idx)
        demo = get_stored_demo(data_path=data_path,
                               index=d_idx,
                               cameras=cameras,
                               depth_scale=depth_scale)

        # get language goal from diskeach
        varation_descs_pkl_file = os.path.join(data_path, episode_folder % d_idx, VARIATION_DESCRIPTIONS_PKL)
        with open(varation_descs_pkl_file, 'rb') as f:
          descs = pickle.load(f)

        # extract keypoints
        # episode_keypoints = _keypoint_discovery(demo, d_idx, stopping_delta) # NOTE: Manually defined keypoints - unused here
        # episode_keypoints = _keypoint_discovery(demo, stopping_delta) # Discover keypoints for current demo index
        episode_keypoints = _keypoint_discovery_available(demo, approach_distance)
        episode_keypoints = episode_keypoints if use_approach else [episode_keypoints[-1]]
        
        # extract (potential) target object locations NOTE: assumed - closed gripper is object location
        episode_target_object = _target_object_discovery(demo, keypoints=target_obj_keypoint, stopping_delta=stopping_delta, last_kp=target_obj_use_last_kp, is_available=target_obj_is_avail)

        num_samples = 5
        prev_keypoint = 0
        is_terminal = False
        
        for i, episode_keypoint in enumerate(episode_keypoints):
             
             if (i == len(episode_keypoints) - 1): # Last checkpoint reached
                  is_terminal = True
             
             # Uniformly create samples between keypoints
             samples = np.linspace(prev_keypoint, episode_keypoint, num=num_samples, endpoint=False, dtype=int)
            
            # Loop through samples inbetween keypoints
             for sample in samples:
                  
                  # Get the observation and task description
                  obs = demo[sample]
                  desc = descs[0]
                  
                  _add_keypoint_to_replay_uniform(
                       replay,
                       obs,
                       demo, episode_keypoint,
                       episode_target_object[sample], #TODO - new check
                       cameras, rlbench_scene_bounds, voxel_sizes, rotation_resolution, crop_augmentation, description=desc,
                       clip_model=clip_model, device=device,
                       episode_index=d_idx, kf_index=i, is_terminal=is_terminal, frame_idx=sample) # New items
                  
             prev_keypoint = episode_keypoint
    print('Replay uniformly filled with demos .')


def fill_replay_copy_with_crop_from_approach(data_path: str,
                episode_folder: str,
                replay: ReplayBuffer,
                # start_idx: int,
                # num_demos: int,
                d_indexes: List[int],
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                rotation_resolution: int,
                crop_augmentation: bool,
                depth_scale,
                use_approach: bool,
                approach_distance: float,
                stopping_delta,
                target_obj_keypoint: bool = False,
                target_obj_use_last_kp: bool = False,
                target_obj_is_avail: bool = False,
                clip_model = None,
                device = 'cpu'):
    """Same fill replay as original, but this clips the filling until the approach phase"""
    print('Filling replay ...')
    for d_idx in d_indexes: #range(start_idx, start_idx+num_demos): # Loops through expert demos
        print("Filling demo %d" % d_idx)
        demo = get_stored_demo(data_path=data_path,
                                index=d_idx,
                               cameras=cameras,
                               depth_scale=depth_scale) # Single episode demo
        
        flag = False

        # get language goal from disk
        varation_descs_pkl_file = os.path.join(data_path, episode_folder % d_idx, VARIATION_DESCRIPTIONS_PKL)
        with open(varation_descs_pkl_file, 'rb') as f:
            descs = pickle.load(f)

        # extract keypoints
        # episode_keypoints = _keypoint_discovery(demo, stopping_delta) # Discover keypoints for current demo index
        episode_keypoints = _keypoint_discovery_available(demo, approach_distance, debug=False)
        if not use_approach:
            raise ValueError("For using this fill replay setting, 'use_approach' must be set to True.")
        episode_keypoints = episode_keypoints if use_approach else [episode_keypoints[-1]] # NOTE: Here we require 2 keyframes, but later we will stop filling at approach keyframe
        print(f'Found {len(episode_keypoints)} keypoints: {episode_keypoints}. Using keypoint:{episode_keypoints[-1]} and sampling till {episode_keypoints[0]}')

        # extract (potential) target object locations NOTE: assumed - closed gripper is object location
        episode_target_object = _target_object_discovery(demo, keypoints=target_obj_keypoint, stopping_delta=stopping_delta, last_kp=target_obj_use_last_kp, is_available=target_obj_is_avail)
        
        for i, (obs, obs_episode_target_object) in enumerate(zip(demo,episode_target_object)): # Loop through frames of demo
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0: # choose only every n-th frame
                continue
            
            # obs = demo[i] # Get the observation at i-th frame
            desc = descs[0]
            target_object = episode_target_object[i]

            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]: # Key-point discovered and frame beyond 1st-keypoint
                flag = True
                break # NOTE: STOP at approach keyframe
                # episode_keypoints = episode_keypoints[1:] # Remove keypoint
            if flag: # NOTE: STOP at approach keyframe
                break
            if len(episode_keypoints) == 0: # No episode key-points discovered
                break
            _add_keypoints_to_replay(
                replay, # Where to store
                obs, # Current observation (i.e. data-X)
                demo, [episode_keypoints[-1]], # (i.e. data-Y) # NOTE: Only use last keypoint 'grasp'
                target_object,
                cameras, rlbench_scene_bounds, voxel_sizes, rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device,
                episode_index=d_idx, frame_idx=i) # New items)
            
    print('Replay filled with demos.')