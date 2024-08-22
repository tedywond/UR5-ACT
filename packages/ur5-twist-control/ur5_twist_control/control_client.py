#!/usr/bin/env python3
from __future__ import annotations

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright 2021 FZI Forschungszentrum Informatik
# Created on behalf of Universal Robots A/S
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -- END LICENSE BLOCK ------------------------------------------------
#
# ---------------------------------------------------------------------
# !\file
#
# \author  Felix Exner mauch@fzi.de
# \date    2021-08-05
#
#
# ---------------------------------------------------------------------
import sys
from typing import List

import actionlib
import geometry_msgs.msg as geometry_msgs
import numpy as np
import rospy
from cartesian_control_msgs.msg import CartesianTrajectoryPoint, FollowCartesianTrajectoryAction, FollowCartesianTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import ListControllers, ListControllersRequest, LoadController, LoadControllerRequest, SwitchController, SwitchControllerRequest
from ros_utils import ROS_INFO, ROS_WARN
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Takuma: This script is taken from
# https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/scripts/test_move


# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

# If your robot description is created with a tf_prefix, those would have to be adapted
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# All of those controllers can be used to execute joint-based trajectories.
# The scaled versions should be preferred over the non-scaled versions.
JOINT_TRAJECTORY_CONTROLLERS = [
    "arm_controller",  # "scaled_pos_joint_traj_controller",  <-- In RIPL we renamed it
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]

# All of those controllers can be used to execute Cartesian trajectories.
# The scaled versions should be preferred over the non-scaled versions.
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]

CARTESIAN_VELOCITY_CONTROLLERS = [
    "twist_controller"
]

# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller", "arm_controller", "forward_cartesian_traj_controller"]


class ControllerManagerClient:
    """Small trajectory client to test a joint trajectory"""

    def __init__(self):
        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "/controller_manager/switch_controller", SwitchController
        )
        self.load_srv = rospy.ServiceProxy("/controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller switch service. Msg: {}".format(err))
            sys.exit(-1)

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        other_controllers = (
            JOINT_TRAJECTORY_CONTROLLERS
            + CARTESIAN_TRAJECTORY_CONTROLLERS
            + CARTESIAN_VELOCITY_CONTROLLERS
            + CONFLICTING_CONTROLLERS
        )

        other_controllers.remove(target_controller)

        srv = ListControllersRequest()
        response = self.list_srv(srv)
        for controller in response.controller:
            if controller.name == target_controller and controller.state == "running":
                rospy.loginfo(f'target controller {target_controller} is already running')
                return

        # Load the target controller
        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        # Start the target controller and turn off the other controllers
        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)


class TwistVelocityClient:
    """Small twist control client"""
    _controller = 'twist_controller'

    def __init__(self, controller_manager: ControllerManagerClient):
        self.controller_manager = controller_manager

        queue_size = rospy.get_param('~twvel_queue_size', 1)
        twvel_rate = rospy.get_param('~twvel_rate', 10)
        self.rate = rospy.Rate(twvel_rate)
        self.vel_publisher = rospy.Publisher('/twist_controller/command', Twist, queue_size=queue_size)
        self.max_lin_vel = 0.4
        self.max_ang_vel = 0.2

    def move(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        # assert linear_vel.shape[0] == 3
        # assert angular_vel.shape[0] == 3

        # if np.linalg.norm(linear_vel) > self.max_lin_vel:
        #     ROS_WARN(f'norm(linear_vel) == {np.linalg.norm(linear_vel)} > 0.2 may be dangerous. Ignoring the input')
        #     return
        # if np.linalg.norm(angular_vel) > self.max_ang_vel:
        #     ROS_WARN(f'norm(angular_vel) == {np.linalg.norm(angular_vel)} > 0.2 may be dangerous. Ignoring the input')
        #     return
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = [float(e) for e in linear_vel]
        twist.angular.x, twist.angular.y, twist.angular.z = [float(e) for e in angular_vel]

        self.vel_publisher.publish(twist)

    def switch_controller(self, target_controller=_controller):
        self.controller_manager.switch_controller(target_controller)


class TrajectoryClient:
    """Small trajectory client to test a joint trajectory"""
    _controller = 'forward_cartesian_traj_controller'

    def __init__(self, controller_manager: ControllerManagerClient):
        self.controller_manager = controller_manager

        self.trajectory_client = actionlib.SimpleActionClient(
            f"/{self._controller}/follow_cartesian_trajectory",
            FollowCartesianTrajectoryAction,
        )
        self.joint_traj_client = actionlib.SimpleActionClient(
            "/arm_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

    def send_cartesian_trajectory(self, trajectory: list[Pose], init_time=3.0, time_step=1.0) -> bool:
        """Creates a Cartesian trajectory and sends it using the selected action server"""

        # make sure the correct controller is loaded and activated
        goal = FollowCartesianTrajectoryGoal()

        rospy.loginfo('controller seems to be correctly loaded!!!!')

        # Wait for action server to be ready
        timeout = rospy.Duration(20)
        if not self.trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        # The following list are arbitrary positions
        # Change to your own needs if desired
        for i, pose in enumerate(trajectory):
            point = CartesianTrajectoryPoint()
            point.pose = pose
            point.time_from_start = rospy.Duration(init_time + time_step * i)
            goal.trajectory.points.append(point)

        rospy.loginfo(f"Executing trajectory using the {self._controller}")
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

        result = self.trajectory_client.get_result()
        if result.error_code != 0:
            rospy.loginfo("Trajectory execution finished in state {}".format(result.error_code))
            rospy.loginfo("error string: {}".format(result.error_string))
            return False

            # rospy.loginfo("invalid joints {}".format(result.INVALID_JOINTS))
            # rospy.loginfo("invalid goal {}".format(result.INVALID_GOAL))
            # rospy.loginfo("invalid posture {}".format(result.INVALID_POSTURE))
        rospy.loginfo('Finished send_cartesian_trajectory!')
        return True

    def send_joint_trajectory(self, trajectory: List[List[float]], init_time=3.0, time_step=1.0):
        """Creates a joint trajectory and sends it using the selected action server"""
        traj_client = self.joint_traj_client

        # make sure the correct controller is loaded and activated
        traj_goal = FollowJointTrajectoryGoal()
        traj_goal.trajectory.joint_names = [
            'ur_arm_elbow_joint', 'ur_arm_shoulder_lift_joint', 'ur_arm_shoulder_pan_joint', 'ur_arm_wrist_1_joint', 'ur_arm_wrist_2_joint', 'ur_arm_wrist_3_joint'
        ]

        traj_goal.trajectory.header = Header()
        traj_goal.trajectory.header.frame_id = 'ur_arm_base'
        for i, jpos in enumerate(trajectory):
            traj_goal.trajectory.points.append(
                JointTrajectoryPoint(
                    positions=jpos,
                    time_from_start=rospy.Duration(init_time + time_step * i)
                )
            )

        rospy.loginfo('controller seems to be correctly loaded!!!!')

        # Wait for action server to be ready
        timeout = rospy.Duration(20)
        if not traj_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        rospy.loginfo(f"Executing trajectory using the {self._controller}")
        traj_client.send_goal(traj_goal)
        traj_client.wait_for_result()

        result = traj_client.get_result()
        if result.error_code != 0:
            rospy.loginfo("Trajectory execution finished in state {}".format(result.error_code))
            rospy.loginfo("error string: {}".format(result.error_string))
            # rospy.loginfo("invalid joints {}".format(result.INVALID_JOINTS))
            # rospy.loginfo("invalid goal {}".format(result.INVALID_GOAL))
            # rospy.loginfo("invalid posture {}".format(result.INVALID_POSTURE))
        rospy.loginfo('Finished send_cartesian_trajectory!')

    def switch_controller(self, target_controller=_controller):
        self.controller_manager.switch_controller(target_controller)


if __name__ == "__main__":
    class Pose:
        def __init__(self, pos, quat):
            self.pos = np.array(pos, dtype=np.float32)
            self.quat = np.array(quat, dtype=np.float32)

        def __repr__(self):
            return repr('<Pose: [{pos}] [{quat}]>'.format(pos=' '.join([f'{val:.2f}' for val in self.pos]), quat=' '.join([f'{val:.2f}' for val in self.quat])))

    rospy.init_node(node_name='cartesian_controller')
    client = TrajectoryClient()
    step = 0.05
    example_trajectory_y = [
        Pose(pos=(0.1, -0.4 + step * i, 0.4), quat=(0.928, -0.371, 0, 0))
        for i in range(4)
    ]
    client.send_cartesian_trajectory(example_trajectory_y)
