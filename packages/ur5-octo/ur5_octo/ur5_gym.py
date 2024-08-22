#!/usr/bin/env python3


import functools
import os
import signal
import subprocess
import time
from time import perf_counter
from copy import deepcopy
from threading import Thread
from collections import deque
from typing import List, Union, Dict

import cv2
import message_filters
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion
# import rosbag
from scipy.spatial.transform import Rotation as R
from ur5_twist_control.control_client import (ControllerManagerClient,
                                              TwistVelocityClient)
from ur5_twist_control.helper import ori2numpy, point2numpy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from robotiq_s_interface.gripper import SModelRobotInput, SModelRobotOutput


class catchtime:
    def __init__(self, desc: str = ''):
        self.desc = desc

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.readout = f'{self.desc} | Time: {self.time:.3f} seconds'
        print(self.readout)


bridge = CvBridge()


class State:
    from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
    def __init__(self):
        self._tf_listener = tf.TransformListener()

    def get_curr_pose(self):
        source_frame = '/hand_finger_tip_link'
        target_frame = '/ur_arm_base'
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        pose = Pose(position=position, orientation=orientation)

        gripper_pos = point2numpy(pose.position)
        gripper_quat = ori2numpy(pose.orientation)
        return gripper_pos, gripper_quat


class Camera:
    def __init__(self, use_compressed_imgs=False, num_stacked_frames=2, cam_fps=30, desired_fps=10, use_grasps=True):
        self.use_grasps = use_grasps
        self.use_compressed_imgs = use_compressed_imgs
        self.num_stacked_frames = num_stacked_frames
        if use_compressed_imgs:
            camarm_topic = '/gripper_cam/color/image_rect_color/compressed'
            camouter_topic = '/outer_cam/color/image_rect_color/compressed'
            _sub_armcam = message_filters.Subscriber(camarm_topic, CompressedImage)
            _sub_outercam = message_filters.Subscriber(camouter_topic, CompressedImage)
        else:
            camarm_topic = '/gripper_cam/color/image_rect_color'
            camouter_topic = '/outer_cam/color/image_rect_color'
            _sub_armcam = message_filters.Subscriber(camarm_topic, Image)
            _sub_outercam = message_filters.Subscriber(camouter_topic, Image)

        # Grasp information
        grasp_fps = 10
        grip_num_past_frames_required = int(grasp_fps / desired_fps) * num_stacked_frames
        self.grip_num_past_frames_required = grip_num_past_frames_required
        self.grip_frame_unit = int(grasp_fps / desired_fps)

        gripper_input_topic = '/UR_1/SModelRobotInput'
        self._sub_gripper_input = rospy.Subscriber(gripper_input_topic, SModelRobotInput, self._gripper_input_callback, queue_size=1)
        self.curr_grasps = deque(maxlen=grip_num_past_frames_required)


        self._ts = message_filters.ApproximateTimeSynchronizer([_sub_armcam, _sub_outercam], queue_size=1, slop=0.05)
        # self._ts = message_filters.ApproximateTimeSynchronizer([_sub_armcam], queue_size=1, slop=0.05)
        self._ts.registerCallback(self._cam_callback)

        # 30 frames per sec == 0.03333 sec per frame
        # Ideal publish freq: 10 times per sec == 0.1 sec per frame
        # We want to retrive 0th frame and 3rd frame (bigger index for older frame)
        cam_num_past_frames_required = int(cam_fps / desired_fps) * num_stacked_frames
        self.camarm_img = deque(maxlen=cam_num_past_frames_required)
        self.camouter_img = deque(maxlen=cam_num_past_frames_required)
        self.cam_num_past_frames_required = cam_num_past_frames_required
        self.cam_frame_unit = int(cam_fps / desired_fps)
        self.prev_stamp = rospy.Time.now()

        # NOTE: Dirty, but I'll also retrieve gripper pose in this class!
        self._tf_listener = tf.TransformListener()
        self.poses = deque(maxlen=cam_num_past_frames_required)

    def _gripper_input_callback(self, input_msg: SModelRobotInput):
        gripper_pos_actual = (input_msg.gPOA + min(input_msg.gPOB, input_msg.gPOC)) / (2*113.)
        self.curr_grasps.append(gripper_pos_actual)

    def _cam_callback(self, arm_msg: Union[Image, CompressedImage], outer_msg: Union[Image, CompressedImage, None] = None):
        self.camarm_img.append(arm_msg)
        self.camouter_img.append(outer_msg)

        pos, quat = self.get_curr_pose()
        self.poses.append(np.concatenate((pos, quat)))

    def fetch_img(self):

        # Wait until the buffers get full
        # while not (len(self.camarm_img) == len(self.camouter_img) == len(self.poses) == self.cam_num_past_frames_required):
        while not (len(self.camouter_img) == len(self.poses) == self.cam_num_past_frames_required):
            print('len(camarm)', len(self.camarm_img), 'len(camouter)', len(self.camouter_img), 'len(poses)', len(self.poses))
            print("Waiting for camera images...")
            time.sleep(0.01)

        if self.use_compressed_imgs:
            np_arr = np.fromstring(self.camarm_img.data, np.uint8)
            armcam_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            np_arr = np.fromstring(self.camouter_img.data, np.uint8)
            outercam_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            raise NotImplementedError('Good luck ;)')
        else:
            # TODO: A better way is to "lock" the shared resources (i.e., camarm_img, camouter_img) ?
            with catchtime('deepcopying the image arrays'):  # 1 ~ 2 msec
                camarm_img = deepcopy(list(self.camarm_img))
                camouter_img = deepcopy(list(self.camouter_img))
                poses = deepcopy(list(self.poses))

            # NOTE: Oldest image first
            images = [camarm_img[-i] for i in reversed(range(1, self.cam_frame_unit * self.num_stacked_frames + 1, self.cam_frame_unit))]
            armcam_img = np.stack([bridge.imgmsg_to_cv2(img, desired_encoding='bgr8') for img in images], axis=0)

            images = [camouter_img[-i] for i in reversed(range(1, self.cam_frame_unit * self.num_stacked_frames + 1, self.cam_frame_unit))]
            outercam_img = np.stack([bridge.imgmsg_to_cv2(img, desired_encoding='bgr8') for img in images], axis=0)

        poses = np.stack(
            [poses[-i] for i in reversed(range(1, self.cam_frame_unit * self.num_stacked_frames + 1, self.cam_frame_unit))]
        ).astype(np.float32)

        grasps = np.stack(
            [self.curr_grasps[-i] for i in reversed(range(1, self.grip_frame_unit * self.num_stacked_frames + 1, self.grip_frame_unit))]
        ).astype(np.float32)
        grasps = grasps[..., None]  # Add an extra dim

        return {'arm': armcam_img, 'outer': outercam_img, 'poses': poses, 'finger_poss': grasps}

    def get_curr_pose(self):
        source_frame = '/ur_arm_tool0_controller'
        target_frame = '/ur_arm_base'
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        pose = Pose(position=position, orientation=orientation)

        gripper_pos = point2numpy(pose.position)
        gripper_quat = ori2numpy(pose.orientation)
        return gripper_pos, gripper_quat


class ShallowGripper:
    def __init__(self, mode='pinch', force=20, num_stacked_frames=2):
        from robotiq_s_interface import Gripper
        # Instantiate Gripper Interface
        self.gripper_mode = mode
        self.gripper_force = force
        print('Initializing Gripper...')
        self.gripper = Gripper(
            gripper_force=force  # This is fed to command.rFRA
        )
        self._initialize_gripper()
        print('Initializing Gripper...done')

    def _gripper_input_callback(self, input_msg: SModelRobotInput):
        self.gripper_pos_actual = (input_msg.gPOA + min(input_msg.gPOB, input_msg.gPOC)) / (2*113.)

    def grasp(self, grip: float):
        assert 0 <= grip <= 1.0
        self.gripper.command.rPRA = int(grip * 255.)
        self.gripper.publish_command(self.gripper.command)

    def _initialize_gripper(self):
        self.gripper.activate()

        if self.gripper_mode == 'basic':
            self.gripper.basic_mode()
        elif self.gripper_mode == 'wide':
            self.gripper.wide_mode()
        elif self.gripper_mode == 'pinch':
            self.gripper.pinch_mode()
        elif self.gripper_mode == 'scissor':
            self.gripper.scissor_mode()
        else:
            raise ValueError(f'unknown mode: {self.gripper_mode}')
        self.gripper.grasp()


def _call_gohomepose_srv():
    from ur5_twist_control.srv import GoHomePose
    print('calling go home pose service!!')
    rospy.wait_for_service('/go_homepose')
    try:
        go_homepose = rospy.ServiceProxy('/go_homepose', GoHomePose)
        result = go_homepose()
        print('result', result)
        return result.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return False


class UR5Gym:
    def __init__(self,
                 init_arm=True,
                 dry_run=False,
                 control_freq=10,
                 num_stacked_frames=2,
                 use_grasps=True
                 ):
        rospy.init_node('ur5_gym')
        self._tf_listener = tf.TransformListener()

        self.init_arm = init_arm
        self.dry_run = dry_run

        self.cam_client = Camera(num_stacked_frames=num_stacked_frames, use_grasps=use_grasps)
        self.state_client = State()

        controller_manager_client = ControllerManagerClient()
        self._twistvel_client = TwistVelocityClient(controller_manager_client)

        self._gripper_client = ShallowGripper()

        self.rospy_rate = rospy.Rate(control_freq)
        self._is_shutting_down = False
        self._curr_pose = None

        self.ensemble_schedule = np.array([0.5, 0.25, 0.15, 0.1])
        self.prev_actions = deque(maxlen=len(self.ensemble_schedule))


    def get_observation(self):
        obss = self.cam_client.fetch_img()

        # arm: (N, 720, 1280, 3)
        # outer: (N, 480, 640, 3)
        print('gripper poses', obss['poses'][0])
        self._curr_pose = obss['poses'][-1]
        return {'camarm': obss['arm'], 'camouter': obss['outer'], 'gripper_poses': obss['poses'], 'finger_poss': obss['finger_poss']}

    def reset(self, gohome=True) -> Dict:
        """Reset the arm to initial location"""
        print('=== reset ===')
        if gohome:
            _call_gohomepose_srv()
        obs = self.get_observation()

        return obs

    def step(self, actions, no_observation=False) -> Union[Dict, None]:
        """Execute the actions and get the observation"""
        assert actions[0].shape[-1] == 7, 'action dim must be 7 (linear + angular + grasp)'

        print(f'====== step (with {len(actions)} actions) =======')

        for action in actions:
            if action[:6].max() > 0.5:
                print('action seems too large... skipping it.')
                continue
            linear_vel, angular_vel, grasp = action[:3], action[3:6], action[-1]

            if not self.dry_run:
                self._twistvel_client.move(linear_vel, angular_vel)
                self._gripper_client.grasp(grasp)
            self.rospy_rate.sleep()

        if not no_observation:
            # Get observation
            obs = self.get_observation()

            return obs
        

    def step_te(self, actions):
        """step with temporal ensemble https://arxiv.org/abs/2304.13705"""
        assert actions[0].shape[-1] == 7, 'action dim must be 7 (linear + angular + grasp)'

        obs = self.get_observation()

        self.prev_actions.appendleft(actions)
        action = np.sum(np.stack([self.prev_actions[i][i] * self.ensemble_schedule[i] for i in range(len(self.prev_actions))]), axis=0)
        print('action', action)

        if action[:6].max() > 0.5:
            print('action seems too large... skipping it.')
            return obs
        
        linear_vel, angular_vel, grasp = action[:3], action[3:6], action[-1]
        self._twistvel_client.move(linear_vel, angular_vel)
        self._gripper_client.grasp(grasp)
        return obs

if __name__ == '__main__':

    ur5_gym = UR5Gym()

    print('=== reset ===')
    obss = ur5_gym.reset()
    # print('camarm', obss['camarm'].shape, obss['camarm'].mean())
    # print('camouter', obss['camouter'].shape, obss['camouter'].mean())

    actions = np.zeros((400, 7))
    actions[:40, 0] = 0.01
    actions[40:80, 0] = -0.01
    actions[80:160, 1] = 0.01
    actions[160:240, 1] = -0.01
    actions[240:320, 2] = 0.01
    actions[320:400, 2] = -0.01

    obss = ur5_gym.step(actions=actions)
    print('camarm', obss['camarm'].shape)
    print('poses', obss['gripper_poses'].shape)

    # Make sure to stop the robot!
    ur5_gym.step(actions=np.zeros((1, 7)))

    rospy.signal_shutdown("Program finished.")
