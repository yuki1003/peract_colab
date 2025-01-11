import time
from datetime import datetime
import json

import numpy as np
import torch

from agent.perceiver_io import PerceiverIO
from agent.peract_agent import PerceiverActorAgent

from notebook_helpers.constants import * # Load global constant variables from constants.py


def build_agent(settings, training=True):

    # SETTINGS
    CAMERAS = settings['cameras']
    RGB_AUGMENTATION = settings['RGB_AUGMENTATION']

    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=VOXEL_SIZES[0],
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        num_latents=NUM_LATENTS,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
    )

    peract_agent = PerceiverActorAgent(
        coordinate_bounds=SCENE_BOUNDS,
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=BATCH_SIZE,
        voxel_size=VOXEL_SIZES[0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=ROTATION_RESOLUTION,
        training_iterations = TRAINING_ITERATIONS,
        lr=LEARNING_RATE,
        lr_scheduler=LR_SCHEDULER,
        num_warmup_steps = NUM_WARMUP_STEPS,
        num_cycles = NUM_CYCLES,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        lambda_weight_l2=0.000001,
        transform_augmentation=TRANSFORM_AUGMENTATION,
        rgb_augmentation=RGB_AUGMENTATION,
        optimizer_type='lamb',
    )
    peract_agent.build(training=training, device=device)

    return peract_agent

def agent_training(settings, peract_agent, train_data_iter, test_data_iter, WORKSPACE_DIR, TASK):

    #___TRAINING___

    LOCAL_FREQ = 10
    SAVE_MODELS = True
    GLOBAL_FREQ = 1000
    calc_test_loss = True

    # Misc
    train_loss = 1e8
    test_loss = 1e8
    general_loss = [1e8,1e8]

    # Directories where to save the best models for train/test
    model_run_time = datetime.now()
    model_save_dir = os.path.join(WORKSPACE_DIR,"outputs", "models", TASK, model_run_time.strftime("%Y-%m-%d_%H-%M"))

    model_save_dir_iter = os.path.join(model_save_dir, "run%d")

    model_save_dir_best_general_iter = os.path.join(model_save_dir_iter, "best_model_general")
    model_save_dir_best_train_iter = os.path.join(model_save_dir_iter, "best_model_train")
    model_save_dir_best_test_iter = os.path.join(model_save_dir_iter, "best_model_test")
    model_save_dir_last_iter = os.path.join(model_save_dir_iter, "last_model")
    metrics_save_path_iter = os.path.join(model_save_dir_iter, "training_metrics.json")  # JSON file to save metrics

    metrics_save_path = os.path.join(model_save_dir, "training_metrics.json")  # JSON file to save metrics
    settings_save_path = os.path.join(model_save_dir, "training_settings.json")

    start_time = time.time()

    # Initialize metrics dictionary
    metrics = {
        "train": [],
        "test": []
    }

    for iteration in range(TRAINING_ITERATIONS):
        
        batch = next(train_data_iter)
        batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
        update_dict = peract_agent.update(iteration, batch) # Here backprop == True: for training reaons, hence training_loss == total_loss

        if iteration % LOCAL_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0

            # Log training metrics
            train_metrics = {
                "iteration": iteration,
                "learning_rate": update_dict['learning_rate'],
                "total_loss": update_dict['total_loss'],
                "trans_loss": update_dict['trans_loss'],
                "rot_loss": update_dict['rot_loss'],
                "col_loss": update_dict['col_loss'],
                "elapsed_time": elapsed_time
            }
            metrics["train"].append(train_metrics)

            if calc_test_loss:
                batch = next(test_data_iter)
                batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
                test_update_dict = peract_agent.update(iteration, batch, backprop=False) # Here backprop == False: for evaluation, hence test_loss == total_loss

                # Log test metrics
                test_metrics = {
                    "iteration": iteration,
                    "total_loss": test_update_dict['total_loss'],
                    "trans_loss": test_update_dict['trans_loss'],
                    "rot_loss": test_update_dict['rot_loss'],
                    "col_loss": test_update_dict['col_loss']
                }
                metrics["test"].append(test_metrics)

                print("Iteration: %d/%d | Learning Rate: %f | Train Loss [tot,trans,rot,col]: [%0.2f, %0.2f, %0.2f, %0.2f] | Test Loss [tot,trans,rot,col]: [%0.2f, %0.2f, %0.2f, %0.2f] | Elapsed Time: %0.2f mins"\
                    % (iteration, TRAINING_ITERATIONS, 
                        update_dict['learning_rate'],
                        update_dict['total_loss'], update_dict['trans_loss'], update_dict['rot_loss'], update_dict['col_loss'],
                        test_update_dict['total_loss'], test_update_dict['trans_loss'], test_update_dict['rot_loss'], test_update_dict['col_loss'], 
                        elapsed_time))
            else:
                print("Iteration: %d/%d | Learning Rate: %f| Train Loss [tot,trans,rot,col]: [%0.2f, %0.2f, %0.2f, %0.2f] | Elapsed Time: %0.2f mins"\
                    % (iteration, TRAINING_ITERATIONS, 
                        update_dict['learning_rate'],
                        update_dict['total_loss'], update_dict['trans_loss'], update_dict['rot_loss'], update_dict['col_loss'],
                        elapsed_time))
                
            if (SAVE_MODELS == True):
                save_model_freq_iter_number = (iteration // GLOBAL_FREQ) * GLOBAL_FREQ

                model_save_dir_general_iteration = model_save_dir_best_general_iter % save_model_freq_iter_number
                model_save_dir_best_train_iteration = model_save_dir_best_train_iter % save_model_freq_iter_number
                model_save_dir_best_test_iteration = model_save_dir_best_test_iter % save_model_freq_iter_number

                # Create directories
                if not os.path.exists(model_save_dir_general_iteration):
                    print(f"Could not find {model_save_dir_general_iteration}, creating directory.")
                    os.makedirs(model_save_dir_general_iteration)
                if not os.path.exists(model_save_dir_best_train_iteration):
                    print(f"Could not find {model_save_dir_best_train_iteration}, creating directory.")
                    os.makedirs(model_save_dir_best_train_iteration)
                if not os.path.exists(model_save_dir_best_test_iteration):
                    print(f"Could not find {model_save_dir_best_test_iteration}, creating directory.")
                    os.makedirs(model_save_dir_best_test_iteration)
                    
                # Only save the best if better
                if update_dict['total_loss'] < general_loss[0] and test_update_dict['total_loss'] < general_loss[1]:
                    print("Saving Best Model - General")
                    general_loss = [update_dict['total_loss'], test_update_dict['total_loss']]
                    peract_agent.save_weights(model_save_dir_general_iteration)
                if update_dict['total_loss'] < train_loss:
                    print("Saving Best Model - Train")
                    train_loss = update_dict['total_loss']
                    peract_agent.save_weights(model_save_dir_best_train_iteration)
                if test_update_dict['total_loss'] < test_loss:
                    print("Saving Best Model - Test")
                    test_loss = test_update_dict['total_loss']
                    peract_agent.save_weights(model_save_dir_best_test_iteration)
            
                if (iteration+LOCAL_FREQ) % GLOBAL_FREQ == 0:# and iteration // GLOBAL_FREQ: #0-500 -> 0, 500-1000 -> 1
                    save_model_freq_iter_number = (iteration // GLOBAL_FREQ) * GLOBAL_FREQ

                    # Save last checkpoint
                    model_save_dir_last_iteration = model_save_dir_last_iter%save_model_freq_iter_number
                    metrics_save_path_iteration = metrics_save_path_iter%save_model_freq_iter_number

                    if not os.path.exists(model_save_dir_last_iteration):
                        print(f"Could not find {model_save_dir_last_iteration}, creating directory.")
                        os.makedirs(model_save_dir_last_iteration)

                    print(f"Saving Model - Last stage: {save_model_freq_iter_number}")
                    peract_agent.save_weights(model_save_dir_last_iteration)

                    # Save metrics to JSON file
                    with open(metrics_save_path_iteration, 'w') as f:
                        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

    # Save metrics to JSON file
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

    # Save training settings to JSON file
    with open(settings_save_path, 'w') as f:
        json.dump(settings, f, indent=4, cls=NumpyEncoder)

    print(f"Training metrics and settings saved to {metrics_save_path} and {settings_save_path}")

    del peract_agent

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)