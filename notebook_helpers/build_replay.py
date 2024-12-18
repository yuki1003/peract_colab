import os
import sys
sys.path = [p for p in sys.path if '/peract/' not in p]
import shutil

import numpy as np
np.bool = np.bool_ # bad trick to fix numpy version issue :(
import clip

from arm.replay_buffer import create_replay, fill_replay, uniform_fill_replay, fill_replay_copy_with_crop_from_approach, fill_replay_only_approach_test

from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

from notebook_helpers.constants import * # Load global constant variables from constants.py


def load_replay_buffer(settings): # Taken from analyse_data.ipynb

    # SETTINGS
    FILL_REPLAY_SETTING = settings['fill_replay_setting']
    CAMERAS = settings['cameras']
    USE_APPROACH = settings['keypoint_approach']
    DEMO_AUGMENTATION_EVERY_N = settings['demo_augm_n']

    # Summary of run properties
    print("\nExperiment Setup")
    print(f"Task: {TASK} - SETUP: {SETUP} - Cameras: {len(CAMERAS)}")
    print("Run Properties")
    print(f"Fill replay setting: {FILL_REPLAY_SETTING} - DEMO_AUGM_N: {DEMO_AUGMENTATION_EVERY_N}")

    #___REPLAY-BUFFER___
    train_replay_storage_dir = os.path.join(WORKSPACE_DIR,'replay_train')
    if os.path.exists(train_replay_storage_dir):
        print(f"Emptying {train_replay_storage_dir}")
        shutil.rmtree(train_replay_storage_dir)
    if not os.path.exists(train_replay_storage_dir):
        print(f"Could not find {train_replay_storage_dir}, creating directory.")
        os.mkdir(train_replay_storage_dir)

    test_replay_storage_dir = os.path.join(WORKSPACE_DIR,'replay_test')
    if os.path.exists(test_replay_storage_dir):
        print(f"Emptying {test_replay_storage_dir}")
        shutil.rmtree(test_replay_storage_dir)
    if not os.path.exists(test_replay_storage_dir):
        print(f"Could not find {test_replay_storage_dir}, creating directory.")
        os.mkdir(test_replay_storage_dir)

    train_replay_buffer = create_replay(batch_size=BATCH_SIZE,
                                        timesteps=1,
                                        save_dir=train_replay_storage_dir,
                                        cameras=CAMERAS,
                                        voxel_sizes=VOXEL_SIZES,
                                        image_size=IMAGE_SIZE,
                                        low_dim_size=LOW_DIM_SIZE)

    test_replay_buffer = create_replay(batch_size=BATCH_SIZE,
                                    timesteps=1,
                                    save_dir=test_replay_storage_dir,
                                    cameras=CAMERAS,
                                    voxel_sizes=VOXEL_SIZES,
                                    image_size=IMAGE_SIZE,
                                    low_dim_size=LOW_DIM_SIZE)

    clip_model, preprocess = clip.load("RN50", device=device) # CLIP-ResNet50

    print("-- Train Buffer --")
    if FILL_REPLAY_SETTING.lower() == "uniform":
        uniform_fill_replay(
            data_path=train_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=train_replay_buffer,
            d_indexes=TRAIN_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    elif FILL_REPLAY_SETTING.lower() == "crop":
        fill_replay_copy_with_crop_from_approach(
            data_path=train_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=train_replay_buffer,
            d_indexes=TRAIN_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    elif FILL_REPLAY_SETTING.lower() == "standard":
        fill_replay_only_approach_test(
        # fill_replay(
            data_path=train_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=train_replay_buffer,
            d_indexes=TRAIN_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    else:
        raise ValueError("Unkown setting for fill replay buffer")

        
    print("-- Test Buffer --")
    if FILL_REPLAY_SETTING.lower() == "uniform":
        uniform_fill_replay(
            data_path=test_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=test_replay_buffer,
            d_indexes=TEST_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    elif FILL_REPLAY_SETTING.lower() == "crop":
        fill_replay_copy_with_crop_from_approach(
            data_path=test_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=test_replay_buffer,
            d_indexes=TEST_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    elif FILL_REPLAY_SETTING.lower() == "standard":
        fill_replay_only_approach_test(
        # fill_replay(
            data_path=test_data_path,
            episode_folder=EPISODE_FOLDER,
            replay=test_replay_buffer,
            d_indexes=TEST_INDEXES,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            depth_scale=DEPTH_SCALE,
            use_approach=USE_APPROACH,
            approach_distance=0.3,
            stopping_delta=STOPPING_DELTA,
            target_obj_keypoint=TARGET_OBJ_KEYPOINTS,
            target_obj_use_last_kp=TARGET_OBJ_USE_LAST_KP,
            target_obj_is_avail=TARGET_OBJ_IS_AVAIL,
            clip_model=clip_model,
            device=device,
            )
    else:
        raise ValueError("Unkown setting for fill replay buffer")


    # delete the CLIP model since we have already extracted language features
    del clip_model

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
    train_dataset = train_wrapped_replay.dataset()
    train_data_iter = iter(train_dataset)

    test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    test_dataset = test_wrapped_replay.dataset()
    test_data_iter = iter(test_dataset)

    return train_data_iter, test_data_iter