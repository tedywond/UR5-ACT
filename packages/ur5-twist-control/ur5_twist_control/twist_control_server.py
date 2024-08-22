#! /usr/bin/env python3

import signal
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ur5_twist_control.control_client import ControllerManagerClient, TrajectoryClient, TwistVelocityClient
from ur5_twist_control.helper import get_arm, get_planning_scene, numpy2quat, ori2numpy, point2numpy
from ur5_twist_control.srv import GoHomePose, GoHomePoseResponse

eps = 1e-3
source_frame = '/hand_finger_tip_link'
target_frame = '/ur_arm_base'

# Sane initial configuration
init_joint_pos = [1.3123908042907715, -1.709191624318258, 1.5454894304275513, -1.1726744810687464, -1.5739596525775355, -0.7679112593280237]
# NOTE: BUG: these init pos and ori are of `ur_arm_tool0_controller` frame! NOT `hand_finger_tip_link` frame
# Initial pose
init_pos = Point(x=0.1, y=-0.4, z=0.4)
# self._init_ori = Quaternion(x=0.928, y=-0.371, z=0, w=0)  # OLD
init_ori = Quaternion(x=0.364, y=-0.931, z=0, w=0)


class TwistControlServer:
    def __init__(self, init_arm=True) -> None:
        # get parameters
        # rospy.init_node('external_twist_control')

        self._tf_listener = tf.TransformListener()

        # self.semaphore = BoundedSemaphore()

        print('Instantiating ControllerManagerClient...')
        controller_manager_client = ControllerManagerClient()

        print('Instantiating move clients...')
        self._traj_client = TrajectoryClient(controller_manager_client)

        # Just to get the joint positions
        from sensor_msgs.msg import JointState
        self.joint_state = None
        self._sub_jpos = rospy.Subscriber('/joint_states', JointState, self._jnt_callback, queue_size=1)
        self._pub_jpos = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)

        # Get current pose and move to the initial pose
        if init_arm:
            self._init_arm(use_joint_controller=True)

        # Define a service
        print('Running go_homepose service...')
        self._gohomepose_srv = rospy.Service('go_homepose', GoHomePose, self._handle_gohomepose)

    def _handle_gohomepose(self, req):
        print('Resetting the arm...')
        success = self._init_arm(use_joint_controller=True)
        return GoHomePoseResponse(success=success)

    def _jnt_callback(self, msg):
        self.joint_state = {'pos': msg.position, 'vel': msg.velocity, 'effort': msg.effort}

    def _get_curr_pose(self) -> Pose:
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        import time
        start = time.perf_counter()
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        elapsed = time.perf_counter() - start
        print(f'>>>>>>>>>>>>> elapsed (lookupTransform): {elapsed:.2f}')
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])

        return Pose(position=position, orientation=orientation)

    def _get_curr_twist(self):
        tooltip_frame = source_frame
        base_frame = target_frame

        twist = self._tf_listener.lookupTwistFull(
            tracking_frame=tooltip_frame,
            observation_frame=base_frame,
            reference_frame=tooltip_frame,
            ref_point=(0, 0, 0),
            reference_point_frame=tooltip_frame,
            time=rospy.Time(0),
            averaging_interval=rospy.Duration(nsecs=int(50 * 1e6))  # 50 ms
        )
        return twist

    def _init_arm(self, spawn_twistvel_controller=True, use_joint_controller=True) -> bool:
        """Move to the initial position"""
        if use_joint_controller:
            # Use scaled_pos_joint_traj_controller (i.e., arm_controller) rather than cartesian ones!
            # Load arm_controller
            self._traj_client.controller_manager.switch_controller('arm_controller')

            # Get current joint pos
            while self.joint_state is None:
                print("Waiting for joint_state to be populated...")
                time.sleep(0.1)

            assert self.joint_state['pos'] is not None
            while np.linalg.norm(np.array(self.joint_state['pos'])) < 1e-6:
                print("Waiting for none-zero joint_state...")
                time.sleep(0.1)

            joint_traj = [self.joint_state['pos'], init_joint_pos]
            print('====== Moving Home ======')
            print('point 0', joint_traj[0])
            print('point 1', joint_traj[1])
            print('====== Moving Home DONE ======')
            success = self._traj_client.send_joint_trajectory(joint_traj, time_step=4.)

            # Switch back to twist_controller!
            self._traj_client.controller_manager.switch_controller('twist_controller')
        else:
            # self._traj_client.controller_manager.switch_controller('forward_cartesian_traj_controller')
            success = self._move_to(init_pos, init_ori, spawn_twistvel_controller=spawn_twistvel_controller)
        return success

    def _move_to(self, pos, ori=None, exec_time=3, spawn_twistvel_controller=True):
        # BUG: This method expects the pos and ori in `ur_arm_tool0_controller` frame! NOT `hand_finger_tip_link` frame

        if ori is None:
            ori = self._init_ori

        print('Loading forward_cartesian_traj_controller...')
        self._traj_client.switch_controller()

        # NOTE: We need this rather than self._get_curr_pose() just because the traj_client expects the pose of ur_arm_tool0_controller.
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, 'ur_arm_tool0_controller', rospy.Time(0))
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        curr_pose = Pose(position=position, orientation=orientation)

        target_pose = Pose(position=pos, orientation=ori)

        # Move to the init location
        print(f'Moving to {pos}, {ori}')

        traj = [curr_pose, target_pose]
        success = self._traj_client.send_cartesian_trajectory(traj, init_time=0.0, time_step=exec_time)

        if spawn_twistvel_controller:
            # Spawn TwistController
            self.twist_vel_client.switch_controller()

        return success


if __name__ == '__main__':
    rospy.init_node('twist_control_server')
    twist_contrl = TwistControlServer()

    # keep spinning
    rospy.spin()
