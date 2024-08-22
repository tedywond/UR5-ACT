#!/usr/bin/env python3
import sys
sys.path.append('../act_mod')
sys.path.append('../aloha')  
import os
from ur5_octo.ur5_gym import UR5Gym
from policy import ACTPolicy, CNNMLPPolicy

import torch
import rospy
import numpy as np
import cv2
from typing import Callable
from threading import Thread
import time
import datetime
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from imitate_episodes import make_policy

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


def run(config, ckpt_name = 'policy_last.ckpt'):
    # model = OctoModel.load_pretrained(f"finetuned/{model_name}", step=step)
    ckpt_path = f'../act_mod/ckpt/{ckpt_name}'
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    ur5_gym = UR5Gym(control_freq=10)
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    policy = make_policy(policy_class, policy_config)

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
