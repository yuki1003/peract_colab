import numpy as np
import torch
from arm.utils import stack_on_channel
from arm.utils import point_to_voxel_index, voxel_index_to_point
from arm.augmentation import apply_se3_augmentation, perturb_se3, quaternion_to_matrix

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def _preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = stack_on_channel(replay_sample['%s_rgb' % n])
        pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])

        rgb = _norm_rgb(rgb)

        obs.append([rgb, pcd]) # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd) # only pointcloud
    return obs, pcds

def pcd_bbox(pose, margin, voxel_size, bounds, bs, device="cpu"):

    bounds = np.array(bounds)

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of gripper pose
    pose_trans = pose[:, :3]
    pose_quat_wxyz = torch.cat((pose[:, 6].unsqueeze(1),
                                pose[:, 3:6]), dim=1)
    pose_rot = quaternion_to_matrix(pose_quat_wxyz)
    pose_4x4_rot = identity_4x4.detach().clone() # Only orientation
    pose_4x4_rot[:, :3, :3] = pose_rot
    pose_4x4 = identity_4x4.detach().clone() # Orientation+location
    pose_4x4[:, :3, :3] = pose_rot
    pose_4x4[:, 0:3, 3] = pose_trans

    # Get the pose voxel location
    pose_indices = point_to_voxel_index(pose[:,:3].cpu().numpy(),
                                        voxel_size,
                                        bounds)
    pose_indices = torch.tensor(pose_indices, device=device) # Gripper pose index ofvoxel space in world frame

    # Create Bounding Box around gripper location w/ margin
    bbox_limits = torch.stack([
        pose_indices - margin,
        pose_indices + margin
    ], dim=-1).clamp(min=0, max=voxel_size) # Ensure within bounds

    x_ranges = [torch.arange(bbox_limits[b, 0, 0], bbox_limits[b, 0, 1], device=device) for b in range(bs)]
    y_ranges = [torch.arange(bbox_limits[b, 1, 0], bbox_limits[b, 1, 1], device=device) for b in range(bs)]
    z_ranges = [torch.arange(bbox_limits[b, 2, 0], bbox_limits[b, 2, 1], device=device) for b in range(bs)]
    voxel_bbox_grid = [torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3) 
                    for x, y, z in zip(x_ranges, y_ranges, z_ranges)]
    
    bounding_box_pcd = np.array([voxel_index_to_point(voxel_bb_grid_b.cpu().numpy(), voxel_size, bounds) 
                        for voxel_bb_grid_b in voxel_bbox_grid])
    bounding_box_pcd = [torch.tensor(bounding_box_pcd, device=device).permute(0,2,1).unsqueeze(-1)]
    bounding_box_pcd = perturb_se3(bounding_box_pcd,
                                    identity_4x4.detach().clone(),
                                    pose_4x4_rot,
                                    pose_4x4,
                                    torch.tensor(bounds,device=device).unsqueeze(0))
    
    return bounding_box_pcd