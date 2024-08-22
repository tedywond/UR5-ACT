#!/usr/bin/env python3

"""This is a simple variant of Gripper, where its open and close value can be adjusted at grasp time"""

import rospy
import roslib  # ; roslib.load_manifest('robotiq_s_model_control')
from robotiq_s_model_articulated_msgs.msg import SModelRobotOutput, SModelRobotInput
from time import sleep, time
from enum import IntEnum
from ros_utils import ROS_INFO, ROS_WARN
from threading import Thread
import numpy as np
import signal
import sys

from .gripper import Gripper, GraspMode, _gripper_interface_controller


class AdjustableGripper(Gripper):
    def __init__(
        self,
        control_topic='/UR_1/SModelRobotOutput',
        feedback_topic='/UR_1/SModelRobotInput',
        gripper_force=10,
        register_for_shutdown_signal=False
    ):
        # initialize interface node (if needed)
        if rospy.get_node_uri() is None:
            self.node_name = 'robotiq_s_interface_node'
            rospy.init_node(self.node_name)
            ROS_INFO( "The Gripper interface was created outside a ROS node, a new node will be initialized!" )
        else:
            ROS_INFO( "The Gripper interface was created within a ROS node, no need to create a new one!" )

        # status of the gripper
        self.status_raw = None
        self.is_shutdown = False
        self.grasp_mode = GraspMode.UNITIALIZED

        # create gripper commands publisher
        self.control_topic = control_topic
        self.pub = rospy.Publisher(control_topic, SModelRobotOutput, queue_size=1)

        # create gripper status subscriber
        self.feedback_topic = feedback_topic
        self.sub = rospy.Subscriber(feedback_topic, SModelRobotInput, self.update_status, queue_size=1)

        self.sub_last_status_message_t = -1

        # create command message
        self.command = SModelRobotOutput()

        # set gripper force
        self.gripper_force = gripper_force
        self.command.rFRA = self.gripper_force

        self.controller_clock_sec = 0.2
        self.pinch_mode_closed_unnorm = 113;

        self._sit_and_wait_status()

        self.controller_thread = Thread( target=_gripper_interface_controller, args=[self,] )
        self.controller_thread.start()

        if register_for_shutdown_signal:
            signal.signal(signal.SIGINT, self.shutdown)

    def is_open(self): #TODO: this does not work in scissor mode because we check the wrong axis
        _open_value_threshold = 0.4

        epsilon = 0.08
        current_fingers_position_norm = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ]) / 255.0
        open_position_norm = np.asarray([
            _open_value_threshold,
            _open_value_threshold,
            _open_value_threshold
        ])
        current_fingers_status = np.asarray([
            self.status_raw.gDTA,
            self.status_raw.gDTB,
            self.status_raw.gDTC
        ])
        #
        if( np.count_nonzero(current_fingers_status-1) == 0 ):
            return True
        for i in range(3):
            # print "Finger #%d: d%r" % ( i+1, abs(current_fingers_position_norm[i]-open_position_norm[i]) )
            if( current_fingers_status[i]==3 and \
                current_fingers_position_norm[i] > open_position_norm[i]+epsilon
                ):
                return False
        return True

    def activate(self):
        self.command = SModelRobotOutput()
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rSPA = 255
        self.command.rFRA = self.gripper_force
        self.publish_command(self.command)
        self._sit_and_wait_activation()
        self.grasp(0.)
        return self

    def is_closed(self): #TODO: this does not work in scissor mode because we check the wrong axis
        _close_value_threshold = 0.6

        epsilon = 0.08
        current_fingers_position_norm = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ]) / 255.0
        closed_position_norm = np.asarray([
            _close_value_threshold,
            _close_value_threshold,
            _close_value_threshold
        ])
        if self.grasp_mode == GraspMode.PINCH:
            closed_position_norm = np.asarray([
                self.pinch_mode_closed_unnorm,
                self.pinch_mode_closed_unnorm,
                self.pinch_mode_closed_unnorm
            ])/255.0
        current_fingers_status = np.asarray([
            self.status_raw.gDTA,
            self.status_raw.gDTB,
            self.status_raw.gDTC
        ])
        #
        if( np.count_nonzero(current_fingers_status-2) == 0 ):
            return True
        for i in range(3):
            # print "Finger #%d: d%r" % ( i+1, abs(current_fingers_position_norm[i]-closed_position_norm[i]) )
            if( current_fingers_status[i]==3 and \
                current_fingers_position_norm[i] < closed_position_norm[i]-epsilon
                ):
                return False
        return True

    def grasp(self, grip_value: float):
        grip_value = float(max(0.0, min(grip_value, 1.0)))
        self.command.rPRA = int(grip_value * 255.0)
        self.publish_command(self.command)
        self._sit_and_wait_grasp()
        return self

    open = NotImplemented
    close = NotImplemented
    release = NotImplemented
