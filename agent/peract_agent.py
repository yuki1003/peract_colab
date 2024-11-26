import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import transformers

from agent.q_function import QFunction
from agent.utils import _preprocess_inputs, pcd_bbox
from agent.voxel_grid import VoxelGrid

from arm.optim.lamb import Lamb
from arm.utils import stack_on_channel
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

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod([self._image_resolution[0], self._image_resolution[1]]) * len(self._camera_names),
        )
        self._vox_grid = vox_grid

        self._q = QFunction(self._perceiver_encoder,
                            vox_grid,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

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
        
        # TODO: Add learning rate
        # learning rate scheduler
        if self._lr_scheduler:
            self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                self._optimizer,
                num_warmup_steps=self._num_warmup_steps,
                num_training_steps=self._training_iterations,
                num_cycles=self._num_cycles,
            )
        else:
            self._scheduler = None

    def _softmax_q(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

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
            gripper_bbox_pcd = pcd_bbox(gripper_pose, 10, self._voxel_size, self._coordinate_bounds, bs)
            object_bbox_pcd = pcd_bbox(object_pose, 10, self._voxel_size, self._coordinate_bounds, bs)

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
            'q_trans': self._softmax_q(q_trans),
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
    
    def predict(self, replay_sample: dict):
        
        # metric scene bounds
        bounds = self._coordinate_bounds

        # inputs
        proprio = stack_on_channel(replay_sample['low_dim_state'])
        obs, pcd = _preprocess_inputs(replay_sample, self._camera_names)
        lang_goal_embs = replay_sample['lang_goal_embs'][:, -1].float()

        # Q function TODO: I think forward of Qfunction
        q_trans, rot_grip_q, collision_q, voxel_grid \
            = self._q(obs,
                      proprio,
                      pcd,
                      lang_goal_embs,
                      bounds)

        raise NotImplementedError
        # return q_trans, rot_grip_q # NOTE: CHECK WHAT FORMAT THIS IS? VOXELS? COORDINATES? etc.

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    print("key %s not found in checkpoint" % k)
        if not self._training:
            # reshape voxelizer weights
            b = merged_state_dict['_voxelizer._ones_max_coords'].shape[0]
            merged_state_dict['_voxelizer._ones_max_coords'] = merged_state_dict['_voxelizer._ones_max_coords'][0:1]
            flat_shape = merged_state_dict['_voxelizer._flat_output'].shape[0]
            merged_state_dict['_voxelizer._flat_output'] = merged_state_dict['_voxelizer._flat_output'][0:flat_shape // b]
            merged_state_dict['_voxelizer._tiled_batch_indices'] = merged_state_dict['_voxelizer._tiled_batch_indices'][0:1]
            merged_state_dict['_voxelizer._index_grid'] = merged_state_dict['_voxelizer._index_grid'][0:1]
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % 'peract_agent'))