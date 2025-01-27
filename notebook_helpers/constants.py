import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import torch

## STATIC VALUES USED IN BELOW FUNCTION: SETTING THEM AS GLOBAL FOR FURTHER USE
#___DATA___
# TASK = 'handing_over_pudding_box'
# # Data Constants
# WORKSPACE_DIR = os.getcwd()
# DATA_FOLDER = os.path.join(WORKSPACE_DIR, "task_data", "handoversim_v3")
# # DATA_FOLDER = "/media/ywatabe/ESD-USB/task_data/handoversim_v3"
# DATA_FOLDER = DATA_FOLDER.replace("/peract_colab", "")
# EPISODES_FOLDER = os.path.join(TASK, "all_variations", "episodes")

# EPISODE_FOLDER = 'episode%d'
# SETUP = "s1" # Options: "s1"
# train_data_path = os.path.join(DATA_FOLDER, f"train_{SETUP}", EPISODES_FOLDER)
# TRAIN_INDEXES = [int(episode_nr.replace("episode", "")) for episode_nr in natsorted(os.listdir(train_data_path))]
# test_data_path = os.path.join(DATA_FOLDER, f"val_{SETUP}", EPISODES_FOLDER)
# TEST_INDEXES = [int(episode_nr.replace("episode", "")) for episode_nr in natsorted(os.listdir(test_data_path))]

# print(f"TRAIN | Total #: {len(TRAIN_INDEXES)}, indices: {TRAIN_INDEXES}")
# print(f"TEST | Total #: {TEST_INDEXES}")

# Replaybuffer related constants
LOW_DIM_SIZE = 4    # 4 dimensions - proprioception: {gripper_open, left_finger_joint, right_finger_joint, timestep}
IMAGE_SIZE =  128  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
# DEMO_AUGMENTATION_EVERY_N = 5#10 NOTE CHANGED through setting # Only select every n-th frame to use for replaybuffer from demo
ROTATION_RESOLUTION = 5 # degree increments per axis
TARGET_OBJ_KEYPOINTS=False # Real - (changed later)
TARGET_OBJ_USE_LAST_KP=False # Real - (changed later)
TARGET_OBJ_IS_AVAIL = True # HandoverSim - (changed later)

DEPTH_SCALE = 1000
STOPPING_DELTA = 0.001
SCENE_BOUNDS = [0.11, -0.5, 0.8, 1.11, 0.5, 1.8]  # Must be 1m each

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training/Validating Settings Constants
BATCH_SIZE = 1 # ANALYSE: 1, TRAINING: 2, INFERENCE: 4-6
TRAINING_ITERATIONS = 7000
LEARNING_RATE = 0.001
TRANSFORM_AUGMENTATION = True

# Unused training settings
LR_SCHEDULER = False
NUM_WARMUP_STEPS = 300  # LR_SCHEDULER: losses seem to stabilize after ~300 iterations
NUM_CYCLES = 1  # As per: https://github.com/peract/peract/blob/02fb87681c5a47be9dbf20141bedb836ee2d3ef9/agents/peract_bc/qattention_peract_bc_agent.py#L232
VOXEL_SIZES = [100]  # 100x100x100 voxels
NUM_LATENTS = 512  # PerceiverIO latents: lower-dimension features of input data