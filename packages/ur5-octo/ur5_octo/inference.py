#!/usr/bin/env python3
import sys
sys.path.append('/code/src/ur5-octo/octo')

import os
from ur5_octo.ur5_gym import UR5Gym
from octo.model.octo_model import OctoModel
import tensorflow_datasets as tfds
import rospy
import numpy as np
import jax
import cv2
from typing import Callable
from threading import Thread
import time
import datetime


def shutdown(ur5_gym):
    print('==================== shutting down =====================')
    ur5_gym._is_shutting_down = True
    import sys
    sys.exit(0)


class CmdlineListener:
    def __init__(self, trigger_fn: Callable, cmd: str = 'q'):
        self.input_thread = Thread(target=self._listener)
        self.input_thread.start()
        self.cmd = cmd
        self.trigger_fn = trigger_fn

    def _listener(self):
        while True:
            command = input("Enter command: ")
            if command == self.cmd:
                print("========== Running trigger_fn() ==============")
                self.trigger_fn()

    def _shutdown(self):
        self.input_thread.join()

def test_action():
    ur5_gym = UR5Gym()
    obss = ur5_gym.reset()

    actions = np.zeros((400, 7))
    actions[:10, 2] = 0.03
    actions[10:20, 2] = -0.03

    obss = ur5_gym.step(actions=actions)

def dry_run(model):
    example_state = np.array([0.09795597, -0.39984968, 0.30922017, 0.7062277, -0.01457974, -0.7076632, -0.01557998, 1.])
    example_state = np.array([[example_state] * 2])
    example_image = cv2.imread('/code/src/ur5-octo/packages/ur5-octo/ur5_octo/example_image.jpg')
    example_image = np.array([[example_image] * 2])

    task = model.create_tasks(texts=['Put the yellow block on the rubik cube.'])

    start = time.time()

    for t in range(100):
        if example_image[0] is None or example_state[0] is None:
            Warning("None in image or state")
        actions = sample(model, example_image, example_state, task)

    end = time.time()
    print(f"Time taken: {end-start}")

def replay():
    sys.path.append('/code/src/ur5-octo/octo')
    from octo.data.dataset import make_single_dataset
    from octo.data.utils.data_utils import NormalizationType

    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="octo_data_may1",
            data_dir="/code/src/ur5-octo/datasets",
            image_obs_keys={"primary": "image"},
            language_key="language_instruction",
            action_proprio_normalization_type=NormalizationType.NORMAL,
            absolute_action_mask=[False] * 6 + [True],
        ),
        traj_transform_kwargs=dict(
            window_size=2,
            future_action_window_size=4,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
            image_augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
                random_contrast=[0.9, 1.1],
                random_saturation=[0.9, 1.1],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
        ),
        train=True,
    )

    iterator = dataset.iterator()
    actions = next(iterator)['action']
    actions = (actions * dataset.dataset_statistics['action']["std"]) + dataset.dataset_statistics['action']["mean"]
    actions[:, :, -1] = np.clip(actions[:, :, -1], a_min=0.0, a_max=1.0)

    ur5_gym = UR5Gym(control_freq=10)
    obss = ur5_gym.reset()
    for i in range(actions.shape[0]):
        obss = ur5_gym.step(actions[i, :1])

def replay_raw_data():
    import pickle
    data = pickle.load(open("/code/src/ur5-octo/rlds_dataset_builder/octo_data_May1/pick_place_May1/65.pt", 'rb'))

    ur5_gym = UR5Gym(control_freq=10)
    obss = ur5_gym.reset()
    for i in range(len(data)):
        action = np.concatenate([data[i]['cmd_trans_vel'], data[i]['cmd_rot_vel'], np.array([data[i]['cmd_grasp_pos']])]).astype(np.float32)
        obss = ur5_gym.step(action[None, :])
        # cv2.imwrite(f'imgs/anno_{i}.jpg', anno_frames)

# This decorator almost doubles the inference frequency
# currently a little below 10Hz
@jax.jit
def sample(model, image_primary=None, image_wrist=None, state=None, task=None):
    observation = {
        'image_primary': image_primary,    # (1, time horizon, H, W, 3)
        'image_wrist': image_wrist,     # (1, time horizon, H, W, 3)
        # 'proprio': state,  # shape: [1, time horizon, 8]
        'pad_mask': np.array([[True] * 2]),  # shape: [1, time horizon]
    }

    actions = model.sample_actions(
        observations=observation,
        tasks=task,
        rng=jax.random.PRNGKey(0)
    )
    return actions

def run(
    model_name = '0611_pretrain_mask_blue_cup',
    step = 80000,
    text = 'blue cup'):


    model = OctoModel.load_pretrained(f"finetuned/{model_name}", step=step)

    ur5_gym = UR5Gym(control_freq=10)

    task = model.create_tasks(texts=[text])

    listener = CmdlineListener(trigger_fn=lambda: shutdown(ur5_gym))

    obss = ur5_gym.reset()

    print('--- entering main loop ---')

    image_primary_list = []
    image_wrist_list = []
    log_actions = []
    gripper_poses = []
    for i in range(10000):
        if ur5_gym._is_shutting_down:
            ur5_gym.step(np.zeros((1, 7,)), no_observation=True)
            rospy.signal_shutdown('aborted!!')
            break

        assert obss['camouter'] is not None
        image_primary = np.array([cv2.resize(obss['camouter'][0], (256, 256)), cv2.resize(obss['camouter'][1], (256, 256))])
        image_primary = np.array([image_primary])
        image_wrist = np.array([cv2.resize(obss['camarm'][0], (128, 128)), cv2.resize(obss['camarm'][1], (128, 128))])
        image_wrist = np.array([image_wrist])


        # state = np.concatenate([obss['gripper_poses'], obss['finger_poss']], axis=1)
        # state = np.array([state])
        # state = (state - model.dataset_statistics['proprio']["mean"]) / (model.dataset_statistics['proprio']["std"] + 1e-8)

        actions = sample(model, image_primary=image_primary, image_wrist=image_wrist, task=task) #, state=state)
        actions = (actions * model.dataset_statistics['action']["std"]) + model.dataset_statistics['action']["mean"]
        actions = np.array(actions)

        actions[:, :, -1] = np.clip(actions[:, :, -1], a_min=0.0, a_max=1.0)  # clip gripper value

        log_actions.append(actions)
        image_primary_list.append(image_primary[0][0])
        image_wrist_list.append(image_wrist[0][0])
        gripper_poses.append(obss['gripper_poses'])

        num_actions_to_run = 4

        # obss = ur5_gym.step(actions[0, :num_actions_to_run])
        obss = ur5_gym.step_te(actions[0])


    print('--- exiting main loop ---')
    log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f'{model_name}_{step}_{log_name}'
    out = cv2.VideoWriter(f'logs/{log_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (512, 256))

    for i in range(len(image_primary_list)):
        image_wrist = cv2.resize(image_wrist_list[i], (256, 256))
        image_primary = image_primary_list[i]
        image = np.concatenate([image_primary, image_wrist], axis=1)
        out.write(image)

    out.release()

    # Kill the command line listener
    listener._shutdown()
    ur5_gym.step(np.zeros((1, 7,)), no_observation=True)
    rospy.signal_shutdown("Program finished.")


if __name__ == "__main__":
    run()
