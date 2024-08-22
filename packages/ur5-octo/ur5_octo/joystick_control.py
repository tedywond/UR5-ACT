#! /usr/bin/env python3
"""
This provides an interface to move the arm with a joystick (TwistController).

- The arm initializes to a specific pose
- A user tries to reach one of the four goals on the table
- Action space is x y z velocity and no orientation
- Bell rings when the x, y location of the tip get inside of the goal
- Buzzer rings when the x, y location of the tip goes out of the goal
- Once the tip reaches the table, play some sound and initialize the arm --> next episode
  - Make sure to record the reached x, y location
"""

import functools
import os
import signal
import subprocess
from copy import deepcopy
from threading import BoundedSemaphore

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion
from robotiq_s_interface import Gripper
# import rosbag
from ros_utils import ROS_INFO, ROS_WARN
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from ur5_twist_control.twist_control import TwistControl
from ur5_twist_reacher.srv import Grasp, GraspResponse, LookupTransform, LookupTransformResponse, MoveTo, MoveToResponse, Reset, ResetResponse, ScriptedPick, ScriptedPickResponse

# NOTE: Make sure that Logitech joystick is in "X" (Xbox?) mode. Not "D" (DirectInput) mode.
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_BACK = 6
BUTTON_START = 7

"""
NOTE: Axes for the black controller of unknown producer
# idx 0: [left pad] -1.0 (left) <--> 1.0 (right)
# idx 1: [left pad] -1.0 (down) <--> 1.0 (up)
# idx 2: [right pad] -1.0 (left) <--> 1.0 (right)
# idx 3: [right pad] -1.0 (down) <--> 1.0 (up)
# idx 4: [left 2] -1.0 (pushed) <--> 1.0 (not pushed)
# idx 5: ?
# idx 6: [left cross] -1.0 (right) <--> 1.0 (left)
# idx 7: [left cross] -1.0 (down) <--> 1.0 (up)

button
idx 7: [left 1]
idx 9: [left 2]
"""

# NOTE: Axes
# idx 0: [left pad] -1.0 (right) <--> 1.0 (left)
# idx 1: [left pad] -1.0 (down) <--> 1.0 (up)
# idx 2: LT
# idx 3: [right pad] -1.0 (right) <--> 1.0 (left)
# idx 4: [right pad] -1.0 (down) <--> 1.0 (up)
# idx 5: RT
# idx 6: [left cross] -1.0 (right) <--> 1.0 (left)
# idx 7: [left cross] -1.0 (down) <--> 1.0 (up)

eps = 1e-3


def point2numpy(point: Point):
    import numpy as np
    return np.array([point.x, point.y, point.z])


class UR5TwistReacher:
    def __init__(self) -> None:
        # get parameters
        rospy.init_node('ur5_twist_reacher')

        self.listener = tf.TransformListener()
        self.source_frame = "ur_arm_tool0_controller"
        self.target_frame = "ur_arm_base"

        self.joy_input_topic = rospy.get_param("~joy_topic", '/joy_teleop/joy')

        self.twist_controller = TwistControl()

        # subscribe to joy commands
        self._sub_joy = rospy.Subscriber(self.joy_input_topic, Joy, self.joy_callback, queue_size=1)

        self._srv_move = rospy.Service('~reset', Reset, self.reset)
        self._srv_tf = rospy.Service('~lookup_tf', LookupTransform, self.lookup_transform)
        self._tf_listener = tf.TransformListener()

        # List of states: ['initializing', 'initialized', 'reaching']
        self.state = 'initializing'
        self.gripper = 0.0

        self.rosbag_recorder = None

    def reset(self, request: Reset):
        ROS_INFO('service -- RESET is called!')
        self.twist_controller._init_arm()
        ROS_INFO('service -- RESET is done!')
        return ResetResponse(success=True)

    def lookup_transform(self, request: LookupTransform):
        """Call tf lookup to retrieve the transformation."""
        source_frame = request.source_frame.data
        target_frame = request.target_frame.data

        timeout = 3.
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        ROS_INFO(f'Looking up transform from {source_frame} to {target_frame}...')
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        ROS_INFO(f'Looking up transform from {source_frame} to {target_frame}...done')
        ROS_INFO(f'trans: {trans}\trot: {rot}')

        pose = Pose(position=Point(*trans), orientation=Quaternion(*rot))
        return LookupTransformResponse(transform=pose)

    def joy_callback(self, msg):
        max_vel = 0.05

        # Left pad
        lp_x, lp_y = msg.axes[0], msg.axes[1]
        rp_x, rp_y = msg.axes[2], msg.axes[3]

        # rospy.loginfo(f'lp: {lp_x:.2f}, {lp_y:.2f}')
        # rospy.loginfo(f'rp: {rp_x:.2f}, {rp_y:.2f}')

        vals = np.array([lp_x, lp_y, rp_y])
        vals[np.abs(vals) < eps] = 0  # Round down small values
        lp_x, lp_y, rp_y = vals

        linear_vel = np.array([lp_x, -lp_y, rp_y], dtype=np.float64) * max_vel
        angular_vel = np.array([0, 0, 0], dtype=np.float64)
        # rospy.loginfo(f'linear_vel: {linear_vel}')
        self.twist_controller.move_vel(linear_vel, angular_vel)

        # Y
        if msg.buttons[4]:
            self.twist_controller._init_arm()

        # right 1
        if msg.buttons[7]:
            self.twist_controller.grasp(0.0)

        # right 2
        if msg.buttons[9]:
            self.twist_controller.grasp(1.0)

        # left 1
        if msg.buttons[6]:
            self.rosbag_recorder = subprocess.Popen(['rosbag', 'record', 'tf', '-O', 'test.bag'])

        # left 2
        if msg.buttons[8] and self.rosbag_recorder is not None:
            self.rosbag_recorder.send_signal(signal.SIGINT)
            self.rosbag_recorder = None

        trans, rot = self.get_relative_transform()
        print(trans, rot)

    # def replay(self):
    #     with rosbag.Bag('test.bag', 'r') as bag:
    #         for topic, msg, t in bag.read_messages():
    #             print("Topic: ", topic)
    #             import pdb; pdb.set_trace()
    #             print("Message: ", msg)
    #             print("Time: ", t)

    def get_relative_transform(self):
        try:
            (trans, rot) = self.listener.lookupTransform(self.target_frame, self.source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.loginfo(f"Error getting transform: {e}")
            return None, None


if __name__ == '__main__':
    ur5_twist_reacher = UR5TwistReacher()
    ROS_INFO('UR5TwistReacher is instantiated')
    ROS_INFO(f'foo {ur5_twist_reacher._srv_move}')

    # keep spinning
    rospy.spin()

    # rosrun ur5_octo joystick_control.py
