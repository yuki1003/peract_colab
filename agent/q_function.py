import copy

import torch
from torch import nn
from torch.nn import DataParallel

import torchvision.transforms as T

# from einops import rearrange, repeat, reduce
# from einops.layers.torch import Reduce

from agent.voxel_grid import VoxelGrid
from arm.utils import point_to_voxel_index
# from arm.network_utils import Conv3DInceptionBlock, DenseBlock, SpatialSoftmax3D, Conv3DInceptionBlockUpsampleBlock, Conv3DBlock, Conv3DUpsampleBlock

class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxel_grid: VoxelGrid,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxel_grid = voxel_grid
        self._qnet = copy.deepcopy(perceiver_encoder)

        # distributed training
        if training:
            self._qnet = DataParallel(self._qnet)
            self._qnet = self._qnet.to(device)

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                obs,
                proprio,
                pcd,
                lang_goal_embs,
                bounds=None,
                backprop=False,
                rgb_augmentation = None,
                gripper_bbox_pcd = None,
                object_bbox_pcd = None):
        # flatten point cloud
        bs = obs[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)

        # flatten rgb
        image_features = [o[0] for o in obs]

        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in
             image_features], 1)

        # voxelize
        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        if backprop and (rgb_augmentation.lower() in ['full', 'partial']):
            transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Create transformation for brightness/contrast/saturation/hue
            ])
            image_features_scaled = torch.cat([(image_features_batch + 1) / 2 for image_features_batch in image_features]) # Conversion [-1,1] -> [0, 1]
            image_features_scaled_augmented = transform(image_features_scaled) # Do transformation
            rgb_unscaled = image_features_scaled_augmented * 2 - 1
            rgb_augmented = [rgb_unscaled[i:i + bs] for i in range(0, len(rgb_unscaled), bs)] # Merge back to batches
            flat_imag_features_augmented = torch.cat(
                [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in rgb_augmented], 1)
            
            voxel_grid_augmented = self._voxel_grid.coords_to_bounding_voxel_grid(pcd_flat,
                                                                        coord_features=flat_imag_features_augmented,
                                                                        coord_bounds=bounds)

            # swap to channels fist
            voxel_grid_augmented = voxel_grid_augmented.permute(0, 4, 1, 2, 3).detach() #(B, (point_robot_cs, RGB, occupancy, position_index), V_depth, V_height, V_width) NOTE: ORDER IS DIFFERENT?
            
            if rgb_augmentation.lower() == 'partial':
                bounding_box_pcd = [torch.cat((gripper_bbox_pcd[0], object_bbox_pcd[0]), dim=2)]
                bounding_box_pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in bounding_box_pcd], 1)
                for b, bounding_box_pcd_flat_b in enumerate(bounding_box_pcd_flat):
                    bounding_box_indices_b = point_to_voxel_index(bounding_box_pcd_flat_b.cpu().numpy(),
                                                                self._voxel_grid._voxel_size,
                                                                bounds[0].cpu().numpy())
                    voxel_grid_augmented[b, 3:6, 
                                        bounding_box_indices_b[:, 0], 
                                        bounding_box_indices_b[:, 1], 
                                        bounding_box_indices_b[:, 2]] = voxel_grid[b, 3:6, 
                                                                                bounding_box_indices_b[:, 0], 
                                                                                bounding_box_indices_b[:, 1], 
                                                                                bounding_box_indices_b[:, 2]]
                
            voxel_grid = voxel_grid_augmented

        # batch bounds if necessary
        if bounds.shape[0] != bs:
            bounds = bounds.repeat(bs, 1)

        # forward pass (TODO: I think forward() of PerceiverIO)
        q_trans, rot_and_grip_q, collision_q = self._qnet(voxel_grid,
                                                          proprio,
                                                          lang_goal_embs,
                                                          bounds)
        return q_trans, rot_and_grip_q, collision_q, voxel_grid

    def latents(self):
        return self._qnet.latent_dict