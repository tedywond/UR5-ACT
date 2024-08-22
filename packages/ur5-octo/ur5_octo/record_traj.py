#!/usr/bin/env python
import rospy
import tf
import numpy as np
import json
from robotiq_s_interface import AdjustableGripper
from ros_utils import ROS_INFO, ROS_WARN
from threading import BoundedSemaphore, Thread
from sensor_msgs.msg import JointState
import os

class DataLogger:
    def __init__(self, source_frame, target_frame, output_file_path, episode):
        rospy.init_node('tf_transform_logger', anonymous=True)
        self.listener = tf.TransformListener()
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.output_file_path = output_file_path

        self.timestep = 0
        self.episode = episode
        self.gripper_state = 0.0
        self.max_timestep = 100
        self.freq = 5
        self.verbose = True
        self.cur_joint_states = None
        self.data = {
            "episode": self.episode,
            "language_instruction": "",
            "freq": self.freq,
            "states": [],
            "joints": [],
            "image_files_arm": [],
            "image_files_side": []
        }

        self.joint_states_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)

        # Start the command line listener thread
        self.input_thread = Thread(target=self.command_line_listener)
        self.input_thread.start()

        self.semaphore = BoundedSemaphore()

        # Instantiate Gripper Interface
        ROS_INFO('Initializing Gripper...')
        self.gripper_force = rospy.get_param("~gripper_force", 10)
        self.gripper = AdjustableGripper(
            gripper_force=self.gripper_force  # This is fed to command.rFRA
        )
        # self.gripper = None
        self.gripper.activate()
        self.gripper.pinch_mode()
        ROS_INFO('Initializing Gripper...done')

    def joint_states_callback(self, msg):
        # Callback for processing joint states
        self.cur_joint_states = msg.position

    def grasp(self, grip: float):
        # return if another message is using the gripper
        gripper_is_free = self.semaphore.acquire(blocking=False)
        if not gripper_is_free:
            ROS_WARN('Gripper is busy...')
            # return GraspResponse(success=False)
            return

        # Open / Close the gripper
        # NOTE: 0: fully open, 1: fully close
        if grip < 0 or 1 < grip:
            ROS_WARN(f'Invalid grip value is recieved: {grip} (expected: 0 <= grip <= 1)')
        else:
            self.gripper.grasp(grip)

        # release lock and exit
        self.semaphore.release()

    def command_line_listener(self):
        print("Command Line Listener Started. Type 'open' to open the gripper or 'close' to close the gripper.")
        while True:
            command = input("Enter command: ")
            if command == 'o':
                self.gripper_state = 0.0
                self.grasp(0.0)
                print("Gripper opend.")
            elif command == 'c':
                self.gripper_state = 1.0
                self.grasp(1.0)
                print("Gripper closed.")
            elif command == 'q':
                rospy.signal_shutdown("User requested shutdown")
                self.save_to_file()
                exit()

    def get_relative_transform(self):
        try:
            (trans, rot) = self.listener.lookupTransform(self.target_frame, self.source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.loginfo(f"Error getting transform: {e}")
            return None, None

    def log_transform(self):
        rate = rospy.Rate(self.freq)  # 1 Hz
        while not rospy.is_shutdown():
            if self.timestep >= self.max_timestep:
                self.save_to_file()
                rospy.signal_shutdown("Max timestep reached")
                exit()
            try:
                self.listener.waitForTransform(self.target_frame, self.source_frame, rospy.Time(0), rospy.Duration(1.0))
                translation, rotation = self.get_relative_transform()
                if self.verbose:
                    print(f"Translation: {translation}, Rotation: {rotation}")
                self.data["states"].append({
                    "timestep": self.timestep,
                    "translation": translation,
                    "rotation": rotation,
                    "gripper_state": self.gripper_state
                })
                self.data["joints"].append({
                    "timestep": self.timestep,
                    "joint_states": self.cur_joint_states
                })
                self.timestep += 1

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.loginfo(f"Error while waiting for transform: {e}")
            rate.sleep()

    def save_to_file(self):
        # make directory if it doesn't exist
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)

        file_name = f"{self.output_file_path}/episode_{self.episode}.json"

        with open(file_name, 'w') as file:
            json.dump(self.data, file)

    def read_from_file(self):
        file_name = f"{self.output_file_path}/episode_{self.episode}.json"
        with open(file_name, 'r') as file:
            data = json.load(file)
            print(data)

if __name__ == "__main__":
    source_link = "ur_arm_tool0_controller"  # Example: "base_link"
    target_link = "ur_arm_base"  # Example: "end_effector_link"
    output_dir = "0215"  # Specify the path to your output file

    # get episode from argument
    import sys
    episode = int(sys.argv[1])

    transform_logger = DataLogger(source_link, target_link, output_dir, episode)
    transform_logger.log_transform()

    # rosrun ur5_octo record_traj.py