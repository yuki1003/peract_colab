import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import clip

import transformers

from agent.q_function import QFunction
from agent.utils import _norm_rgb, _point_cloud_from_depth_and_camera_params, _preprocess_inputs, pcd_bbox
from agent.voxel_grid import VoxelGrid

from arm.optim.lamb import Lamb
from arm.replay_buffer import _clip_encode_text
from arm.utils import discrete_euler_to_quaternion, stack_on_channel
from arm.augmentation import apply_se3_augmentation, perturb_se3


class PerceiverActorAgent():
    def __init__(self,
                coordinate_bounds: list,
                perceiver_encoder: nn.Module,
                camera_names: list,
                batch_size: int,
                voxel_size: int,
                voxel_feature_size: int,
                num_rotation_classes: int,
                rotation_resolution: float,
                training_iterations: int = 10000,
                lr: float = 0.0001,
                lr_scheduler: bool = False,
                num_warmup_steps: int = 2000,
                num_cycles: int = 5,
                image_resolution: list = None,
                lambda_weight_l2: float = 0.0,
                transform_augmentation: bool = True,
                transform_augmentation_xyz: list = [0.125, 0.125, 0.125],
                transform_augmentation_rpy: list = [0.0, 0.0, 45.0], # was [0.0, 0.0, 45.0]
                transform_augmentation_rot_resolution: int = 5,
                rgb_augmentation: bool = None,
                optimizer_type: str = 'lamb'):

        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._training_iterations = training_iterations
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._num_warmup_steps = num_warmup_steps
        self._num_cycles = num_cycles
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = transform_augmentation_xyz
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._rgb_augmentation = rgb_augmentation
        self._optimizer_type = optimizer_type

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def build(self, training: bool, device: torch.device = None, language_goal = None):
        self._training = training
        self._device = device

        self._vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=self._device,
            batch_size=self._batch_size,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod([self._image_resolution[0], self._image_resolution[1]]) * len(self._camera_names),
        )

        self._q = QFunction(self._perceiver_encoder,
                            self._vox_grid,
                            self._rotation_resolution,
                            self._device,
                            training).to(self._device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=self._device).unsqueeze(0)

        if self._training:
            if self._optimizer_type == 'lamb':
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer')
            
            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._num_cycles,
                )

        else:
            for param in self._q.parameters():
                param.requires_grad = False

            if language_goal:
                self._language_goal_embeddings = self.set_language_goal(language_goal)
            else:
                print("Language goal is not set. Use .set_language_goal() to set language goal")

            self._vox_grid.to(self._device)
            self._q.to(self._device)

    def set_language_goal(self, language_goal: str):

        # load CLIP for encoding language goals during evaluation
        language_tokens = clip.tokenize([language_goal]).numpy()
        language_tokens_tensor = torch.from_numpy(language_tokens).to(self._device)
        
        clip_model, _ = clip.load("RN50", device=self._device) # CLIP-ResNet50
        _, lang_embs = _clip_encode_text(clip_model, language_tokens_tensor)
        del clip_model

        language_goal_embeddings = lang_embs.float().detach()

        return language_goal_embeddings
        

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)
    
    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def _get_one_hot_expert_actions(self,  # You don't really need this function since GT labels are already in the right format. This is some leftover code from my experiments with label smoothing.
                                    batch_size,
                                    action_trans,
                                    action_rot_grip,
                                    action_ignore_collisions,
                                    device):
        bs = batch_size

        # initialize with zero tensors
        action_trans_one_hot = torch.zeros((bs, self._voxel_size, self._voxel_size, self._voxel_size), dtype=int, device=device)
        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot  = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            # translation
            gt_coord = action_trans[b, :]
            action_trans_one_hot[b, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

            # rotation
            gt_rot_grip = action_rot_grip[b, :]
            action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
            action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
            action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
            action_grip_one_hot[b, gt_rot_grip[3]] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        # flatten trans
        action_trans_one_hot = action_trans_one_hot.view(bs, -1)

        return action_trans_one_hot, \
               action_rot_x_one_hot, \
               action_rot_y_one_hot, \
               action_rot_z_one_hot, \
               action_grip_one_hot,  \
               action_collision_one_hot


    def update(self, step: int, replay_sample: dict, backprop: bool = True) -> dict:
        # sample
        action_trans = replay_sample['trans_action_indicies'][:, -1, :3].int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1].int()
        action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_gripper_pose = replay_sample['gripper_pose'][:, -1]
        gripper_pose = replay_sample['gripper_state'][:, -1]
        object_pose = replay_sample['object_state'][:, -1]
        lang_goal_embs = replay_sample['lang_goal_embs'][:, -1].float()

        # metric scene bounds
        bounds = self._coordinate_bounds
        
        bs = self._batch_size

        # inputs
        proprio = stack_on_channel(replay_sample['low_dim_state'])
        obs, pcd = _preprocess_inputs(replay_sample, self._camera_names)

        gripper_bbox_pcd = None
        object_bbox_pcd = None
        if backprop and (self._rgb_augmentation.lower() == 'partial'):
            gripper_bbox_pcd = pcd_bbox(gripper_pose, 10, self._voxel_size, self._coordinate_bounds, bs, self._device)
            object_bbox_pcd = pcd_bbox(object_pose, 10, self._voxel_size, self._coordinate_bounds, bs, self._device)

        # SE(3) augmentation of point clouds and actions
        if backprop and self._transform_augmentation:
            action_trans, action_rot_grip, pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4 \
                = apply_se3_augmentation(pcd,
                                         action_gripper_pose, action_trans, action_rot_grip,
                                         bounds,
                                         0,
                                         self._transform_augmentation_xyz, self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size, self._rotation_resolution,
                                         self._device)
            
            if backprop and (self._rgb_augmentation.lower() == 'partial'):
                gripper_bbox_pcd = perturb_se3(gripper_bbox_pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
                object_bbox_pcd = perturb_se3(object_bbox_pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

        # Q function TODO: I think forward of Qfunction
        q_trans, rot_grip_q, collision_q, voxel_grid \
            = self._q(obs,
                      proprio,
                      pcd,
                      lang_goal_embs,
                      bounds,
                      backprop,
                      self._rgb_augmentation,
                      gripper_bbox_pcd, object_bbox_pcd) # NOTE: This is the prediction


        # one-hot expert actions
        action_trans_one_hot, action_rot_x_one_hot, \
        action_rot_y_one_hot, action_rot_z_one_hot, \
        action_grip_one_hot, action_collision_one_hot = self._get_one_hot_expert_actions(bs,
                                                                                         action_trans,
                                                                                         action_rot_grip,
                                                                                         action_ignore_collisions,
                                                                                         device=self._device)
        total_loss = 0.
        # if backprop: CHANGE: 167-188 TODO: Different?
        # cross-entropy loss
        trans_loss = self._cross_entropy_loss(q_trans.view(bs, -1),
                                                action_trans_one_hot.argmax(-1))

        rot_grip_loss = 0.
        rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 0*self._num_rotation_classes:1*self._num_rotation_classes],
                                                    action_rot_x_one_hot.argmax(-1))
        rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 1*self._num_rotation_classes:2*self._num_rotation_classes],
                                                    action_rot_y_one_hot.argmax(-1))
        rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 2*self._num_rotation_classes:3*self._num_rotation_classes],
                                                    action_rot_z_one_hot.argmax(-1))
        rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 3*self._num_rotation_classes:],
                                                    action_grip_one_hot.argmax(-1))

        collision_loss = self._cross_entropy_loss(collision_q,
                                                    action_collision_one_hot.argmax(-1))

        total_loss = trans_loss + rot_grip_loss + collision_loss
        total_loss = total_loss.mean()

        learning_rate = self._lr
        if backprop: # TODO: Add learning rate
            # backprop
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            
            if self._lr_scheduler:
                self._scheduler.step()
                learning_rate = self._scheduler.get_last_lr()[0]

        total_loss = total_loss.item()

        
        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)

        # discrete to continuous translation action
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2

        return {
            'learning_rate': learning_rate,
            'total_loss': total_loss,
            'trans_loss': trans_loss.mean().detach().cpu().numpy(),
            'rot_loss': rot_grip_loss.mean().detach().cpu().numpy(),
            'col_loss': collision_loss.mean().detach().cpu().numpy(),
            'voxel_grid': voxel_grid,
            'q_trans': self._softmax_q_trans(q_trans),
            'pred_action': {
                'trans': coords_indicies,
                'continuous_trans': continuous_trans,
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }
    
    def forward(self, observation, timestep):
        """
        Processes an observation for inference by preparing RGB and point cloud data 
        and constructing inputs for the model.

        Args:
            observation (dict): 
                A dictionary containing observation data with the following keys:
                - "{camera_name}_rgb": RGB images from each camera (shape: [C, H, W]). E.g. (3, 128, 128)
                - "{camera_name}_depth": Depth maps from each camera (shape: [1, H, W]). E.g. (1, 128, 128)
                - "{camera_name}_camera_extrinsics": Camera extrinsic parameters (4x4 matrix).
                - "{camera_name}_camera_intrinsics": Camera intrinsic parameters (3x3 matrix).
                - "gripper_open": Whether the gripper is open (float or bool).
                - "gripper_joint_positions": Joint positions of the gripper (array of floats).
            timestep (int): 
                Current timestep in the episode (0-indexed).
        """

        for camera_name in self._camera_names:
            # Expand dimensions for batch processing
            observation[f"{camera_name}_rgb"] = np.expand_dims(observation[f"{camera_name}_rgb"], axis=0)
            # print(observation[f"{camera_name}_depth"])
            # Compute point cloud from depth and camera parameters
            point_cloud = _point_cloud_from_depth_and_camera_params(
                observation[f"{camera_name}_depth"][0],
                observation[f"{camera_name}_camera_extrinsics"],
                observation[f"{camera_name}_camera_intrinsics"]
            )
            observation[f"{camera_name}_point_cloud"] = np.expand_dims(point_cloud, axis=0).transpose(0, 3, 1, 2)

        # Convert to torch tensors
        obs_dict = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in observation.items()}

        # Inputs: Proprio
        episode_length = 10  # NOTE: Adjust or parameterize if needed
        normalized_time = (1.0 - (timestep / float(episode_length - 1))) * 2.0 - 1.0

        proprio_input = np.array([obs_dict["gripper_open"], *obs_dict["gripper_joint_positions"], normalized_time])
        proprio = torch.from_numpy(proprio_input[None]).to(self._device).float()

        # Put on Device
        obs_dict = {k: v.to(self._device) for k, v in obs_dict.items() if type(v) == torch.Tensor}

        # Inputs: obs and pcd
        obs, pcd = [], []
        for n in self._camera_names:
            rgb_n = obs_dict['%s_rgb' % n].float()
            pcd_n = obs_dict['%s_point_cloud' % n].float()

            rgb_n = _norm_rgb(rgb_n)

            obs.append([rgb_n, pcd_n]) # obs contains both rgb and pointcloud (used in ARM for other baselines)
            pcd.append(pcd_n) # only pointcloud

        # metric scene bounds
        bounds = self._coordinate_bounds

        # Inference - forward of Qfunction
        q_trans, q_rot_grip, collision_q, voxel_grid \
            = self._q(obs,
                      proprio, # Cehck above
                      pcd, # This is the input
                      self._language_goal_embeddings, # Static
                      bounds) # Set before
        
        # softmax Q predictions Found
        # print("Before softmax:", q_trans)
        q_trans = self._softmax_q_trans(q_trans)
        rot_grip_q =  self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(collision_q) \
            if collision_q is not None else collision_q
        # print("After softmax:", q_trans)

        # argmax Q predictions
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          q_ignore_collisions)

        # discrete to continuous translation action
        # print(coords_indicies)
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2
        # print(res, bounds, self._voxel_size)
        # print(continuous_trans)
        continuous_trans = continuous_trans[0].cpu().numpy()
        
        continuous_quat = discrete_euler_to_quaternion(rot_and_grip_indicies[0][:3].detach().cpu().numpy(),
                                                       resolution=self._rotation_resolution)
        gripper_open = bool(rot_and_grip_indicies[0][-1].detach().cpu().numpy())
        ignore_collision = bool(ignore_collision_indicies[0][0].detach().cpu().numpy()) # Check if this is necessary

        return (continuous_trans, continuous_quat, gripper_open, q_trans.max().item(), rot_grip_q.max().item()),    \
              (voxel_grid, coords_indicies, rot_and_grip_indicies, gripper_open)

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, 'peract_agent.pt')
        state_dict = torch.load(weight_file, map_location=device, weights_only=True)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    print("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % 'peract_agent'))