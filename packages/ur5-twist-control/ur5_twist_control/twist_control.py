#! /usr/bin/env python3

import signal
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
from robotiq_s_interface import Gripper
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


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0

    def update(self, error, dt):
        derivative = (error - self.last_error) / dt
        self.integral += error * dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return output


class TwistControl:
    def __init__(self, init_arm=True) -> None:
        # get parameters
        # rospy.init_node('external_twist_control')

        # NOTE: No idea why ~joy_topic is not found.
        self.input_topic = rospy.get_param("~joy_topic", '/external_control')

        # Choices: ['basic', 'wide', 'pinch', 'scissor']
        self.gripper_mode = rospy.get_param("~gripper_mode", "pinch")
        self.gripper_force = rospy.get_param("~gripper_force", 20)

        self._tf_listener = tf.TransformListener()

        # self.semaphore = BoundedSemaphore()

        # Instantiate Gripper Interface
        print('Initializing Gripper...')
        self.gripper = Gripper(
            gripper_force=self.gripper_force  # This is fed to command.rFRA
        )
        self._initialize_gripper()
        print('Initializing Gripper...done')

        print('Instantiating ControllerManagerClient...')
        controller_manager_client = ControllerManagerClient()

        print('Instantiating move clients...')
        self._traj_client = TrajectoryClient(controller_manager_client)
        self.twist_vel_client = TwistVelocityClient(controller_manager_client)

        # Just to get the joint positions
        from sensor_msgs.msg import JointState
        self.joint_state = None
        self._sub_jpos = rospy.Subscriber('/joint_states', JointState, self._jnt_callback, queue_size=1)
        self._pub_jpos = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)

        # Get current pose and move to the initial pose
        if init_arm:
            self._init_arm(use_joint_controller=True)

        # listen for SIGINT
        signal.signal(signal.SIGINT, self._shutdown)

        self.log = []
        # self._prev_vel = (np.zeros(3), np.zeros(3))

        self.pid_controller = PID(0.6, 0.01, 0.01)
        self.pid_controller_angular = PID(0.5, 0.01, 0.01)
        self.prev_angular_vel = np.zeros(3)

        # Define a service
        print('Running go_homepose service...')
        self._gohomepose_srv = rospy.Service('go_homepose', GoHomePose, self._handle_gohomepose)

    def _handle_gohomepose(self, req):
        # TODO
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

    def move_vel(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        with np.printoptions(precision=2, suppress=True):
            print(f'Moving with linear_vel: {linear_vel}, angular_vel: {angular_vel}')
        self.twist_vel_client.move(linear_vel, angular_vel)

    def move_toward(self, pos: np.ndarray, quat: np.ndarray = None, dry_run: bool = False):
        """Based on the current diff in pose, compute a velocity with P controller.
        quat: (x, y, z, w)

        NOTE: Default orientation of the tooltip (looking down):
        (np.pi, 0, -0.76)

        Default orientation of the LMC hand:
        (0, -np.pi / 2, 0)
        """
        curr_pose = self._get_curr_pose()
        curr_pos = point2numpy(curr_pose.position)
        curr_quat = ori2numpy(curr_pose.orientation)

        print("==========================")
        print(f"target pos: {pos}")

        d_pos = pos - curr_pos

        linear_vel = self.pid_controller.update(d_pos, 0.1)

        angular_vel = np.zeros(3)

        if quat is not None:
            # NOTE: quat is in base frame!
            # The input (twist) needs to be in the gripper frame!!
            R_base_grp = R.from_quat(curr_quat)
            R_grp_base = R_base_grp.inv()
            R_base_hand = R.from_quat(quat)  # Target ori in the base frame

            # This is in hand_finger_tip_link frame
            Rtgt_grp_hand = R_grp_base * R_base_hand
            angular_vel = R_base_grp.as_matrix() @ Rtgt_grp_hand.as_rotvec()

            # angular_vel = angular_vel * 0.5
            print(f'angular_vel (before): {np.round(angular_vel, 3)}')
            print('\t\tNORM ANGULAR VEL (1)', round(np.linalg.norm(angular_vel), 3))
            angular_vel = self.pid_controller_angular.update(angular_vel, 0.1)
            angular_vel = np.clip(angular_vel, -0.2, 0.2)
            print('\t\tNORM ANGULAR VEL (2)', round(np.linalg.norm(angular_vel), 3))

            # angular_vel = Rtgt_grp_hand.as_rotvec()
            # angular_vel = Rtgt_grp_hand.as_euler('xyz')

            # trans, rot = self._tf_listener.lookupTransform(
            #     '/ur_arm_tool0_controller',
            #     '/hand_finger_tip_link',
            #     rospy.Time(0))
            # R_tool0_fingertip = R.from_quat(rot)

            # print(f'angular_vel (before): {Rtgt_grp_hand.as_euler("xyz")}')
            # # Convert this rot diff (angular velocity) to ur_arm_tool0_controller
            # angular_vel = R_tool0_fingertip.as_matrix() @ Rtgt_grp_hand.as_rotvec()
            # angular_vel = angular_vel * p_angular_coef=

            angular_acc = angular_vel - self.prev_angular_vel
            # print(f'angular_acc: {np.round(angular_acc, 3)}')

            # angular_acc_bool = np.abs(angular_acc) > 0.25
            # angular_vel[angular_acc_bool] = 0
            angular_acc = np.clip(angular_acc, -0.05, 0.05)
            angular_vel = self.prev_angular_vel + angular_acc
            self.prev_angular_vel = angular_vel
            # print(f'angular_vel (after): {np.round(angular_vel, 3)}')

            print('\t\tNORM ANGULAR VEL (3)', round(np.linalg.norm(angular_vel), 3))
            print('\t\tNORM ANGULAR ACC', round(np.linalg.norm(angular_acc), 3))

            # print(f'angular_acc: {np.round(angular_acc, 3)}')
            # if np.linalg.norm(angular_acc) > 0.1:
            #     angular_vel = angular_vel_before

            # max_angular_vel = 0.05
            max_angular_vel = None  # TEMP
            # print(f'curr ori {R.from_quat(curr_quat).as_euler("xyz")}')
            # print(f'orig_target ori {R.from_quat(quat).as_euler("xyz")}')
            # d_ori = Rtgt_grp_hand
            # angular_vel = d_ori.as_euler('xyz') * p_angular_coef

            angular_vel_norm = max(np.linalg.norm(angular_vel), 0.0001)
            if max_angular_vel is not None and angular_vel_norm > max_angular_vel:
                print(f'norm(angular_vel) = {angular_vel_norm:.3f} > {max_angular_vel:.3f}. Normalizing the angular velocity')
                angular_vel = angular_vel / angular_vel_norm * max_angular_vel

        max_vel = 0.2  # I guess we can go up to like 30cm / sec
        # max_vel = None
        linear_vel_norm = max(np.linalg.norm(linear_vel), 0.0001)
        if max_vel is not None and linear_vel_norm > max_vel:
            # ROS_WARN(f'norm(linear_vel) = {linear_vel_norm:.3f} > {max_vel:.3f}. Normalizing the velocity')
            print(f'norm(linear_vel) = {linear_vel_norm:.3f} > {max_vel:.3f}. Normalizing the velocity')
            linear_vel = linear_vel / linear_vel_norm * max_vel

        curr_vel_trans, curr_vel_rot = self._get_curr_twist()
        # curr_vel_trans, _ = self._prev_vel
        curr_vel_trans = np.array(curr_vel_trans)

        # TODO: We actually want current (measured) velocity rather than the commanded prev_vel
        max_acc = 0.02
        # max_acc = None
        # prev_linear_vel, _ = self._prev_vel
        # print(f'prev_vel vs observed vel: {np.linalg.norm(prev_linear_vel - curr_vel_trans):.4f}')
        # prev_linear_vel = curr_vel_trans

        norm_acc = np.linalg.norm(linear_vel - curr_vel_trans)  # Acceleration

        # prev_vel + a * (new_vel - prev_vel)
        # = (1 - a) * prev_vel + a * new_vel
        # -> norm -> bounded by 0.08
        # Find a such that
        # 1. the norm is less than 0.08
        # 2. as close as possible to new_vel

        if max_acc is not None and norm_acc > max_acc:
            # ROS_WARN(f'norm_acc = {norm_acc:.3f} > {max_acc:.3f}. Normalizing the acceleration')
            print(f'>> norm_acc = {norm_acc:.3f} > {max_acc:.3f}. Normalizing the acceleration')
            linear_vel = curr_vel_trans + max_acc * (linear_vel - curr_vel_trans) / norm_acc

        print(f'norm_acc: {norm_acc:.3f}')
        # linear_vel = prev_linear_vel + min(norm_acc, 0.1) * (linear_vel - prev_linear_vel)

        # print(f'linear_vel: {linear_vel}\tangular_vel: {angular_vel}')

        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'hand_finger_tip_link'
        # quaternion_stamped = QuaternionStamped(
        #     header=header,
        #     quaternion=numpy2quat(quat)
        # )
        # target_frame = 'ur_arm_tool0_controller'
        # quat_in_tool0 = self._tf_transformer.transformQuaternion(target_frame, quaternion_stamped)
        # quat_in_tool0 = self._tf_listener.transformQuaternion(target_frame, quaternion_stamped)
        # angular_vel = R.from_quat(ori2numpy(quat_in_tool0.quaternion)).as_euler('xyz')
        # linear_vel *= 0  # TEMP
        # print(f'angular_vel (after): {angular_vel}')

        # TODO: Convert angular velocity from hand_finger_tip_link to arm_tool0_controller

        if not dry_run:
            self.move_vel(linear_vel, angular_vel)

    def _shutdown(self, *args):
        print('Shutting down...')
        self.gripper.shutdown()


if __name__ == '__main__':
    rospy.init_node('twist_control')
    twist_contrl = TwistControl()

    # keep spinning
    rospy.spin()
