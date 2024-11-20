# From https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py

import numpy as np
from typing import List

from rlbench.demo import Demo


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = ( # NOTE: Verify by 2
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and #NOTE: Current and Next
             obs.gripper_open == demo[i - 1].gripper_open and # NOTE: Current and Prev
             demo[i - 2].gripper_open == demo[i - 1].gripper_open)) # NOTE: PrevPrev and Prev
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta) # Near-0 joint velocities
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo: Demo,
                        stopping_delta=0.1,
                        debug: bool = True) -> List[int]: #NOTE: originally 0.1, but for inspect real data (and its task) to elucidate better stopping delta
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open #NOTE: 1.0==open, <1.0==closed
    stopped_buffer = 0 # skip n frames after a kf
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    if debug:
        print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def _target_object_discovery(demo: Demo, keypoints: bool = False, stopping_delta=0.1, last_kp: bool = False) -> List:
    """Add target object is gripper is closed (i.e. obs.gripper_pose < 1.0), OR use keypoints as reference
    if last_kp == True, then use last keyframe as target object (for simple tasks)
    """

    episode_target_object = []

    if not keypoints:
        gripper_location = demo[-1].gripper_pose
        for i, obs in reversed(list(enumerate(demo))):
            # if change in gripper, or end of episode.
            last_kp = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open < 1.0 or last_kp):
                episode_target_object.append(obs.gripper_pose)
                gripper_location = obs.gripper_pose
            else:
                episode_target_object.append(gripper_location)
        episode_target_object.reverse()
    else:
        episode_keypoints = _keypoint_discovery(demo, stopping_delta, debug=False)
        keypoints_idx = 0
        if last_kp:
            episode_target_object = [demo[episode_keypoints[-1]].gripper_pose for _ in demo]
        if not last_kp:
            for i in range(len(demo)):
                if episode_keypoints[keypoints_idx] > i:
                    keypoints_idx += 1
                target_object = demo[episode_keypoints[keypoints_idx]].gripper_pose
                episode_target_object.append(target_object)

    return episode_target_object