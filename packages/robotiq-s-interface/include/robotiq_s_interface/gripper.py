import rospy
import roslib; roslib.load_manifest('robotiq_s_model_control')
from robotiq_s_model_articulated_msgs.msg import SModelRobotOutput, SModelRobotInput
from time import sleep, time
from enum import IntEnum
from ros_utils import *
from threading import Thread
import numpy as np
import signal
import sys


class GraspMode(IntEnum):
    UNITIALIZED = -1,
    BASIC = 0,
    PINCH = 1,
    WIDE = 2,
    SCISSOR = 3,
    TRANSITIONING = 4


class Gripper():
    def __init__(
        self,
        control_topic='/UR_1/SModelRobotOutput',
        feedback_topic='/UR_1/SModelRobotInput',
        gripper_force=10,
        open_value=0.0,
        close_value=1.0,
        register_for_shutdown_signal=False
        ):

        # initialize interface node (if needed)
        if rospy.get_node_uri() is None:
            self.node_name = 'robotiq_s_interface_node'
            rospy.init_node(self.node_name)
            ROS_INFO( "The Gripper interface was created outside a ROS node, a new node will be initialized!" )
        else:
            ROS_INFO( "The Gripper interface was created within a ROS node, no need to create a new one!" )

        # parameters
        self._open_value = float( max( 0.0, min(open_value, 1.0) ) )
        self._close_value = float( max( 0.0, min(close_value, 1.0) ) )

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


    def reset_controller(self):
        # initialize gripper node
        rospy.init_node(self.node_name)
        # unsubscribe and unregister publishers and subscribers
        self.sub.unsubscribe()
        self.pub.unregister()
        # create gripper commands publisher
        self.control_topic = control_topic
        self.pub = rospy.Publisher(control_topic, SModelRobotOutput, queue_size=1)
        # create gripper status subscriber
        self.feedback_topic = feedback_topic
        self.sub = rospy.Subscriber(feedback_topic, SModelRobotInput, self.update_status, queue_size=1)


    # keep the internal status of this controller class updated
    def update_status(self, msg):
        self.status_raw = msg
        self.sub_last_status_message_t = time()
        if self.grasp_mode != GraspMode.TRANSITIONING:
            self.grasp_mode = GraspMode( msg.gMOD )

    def _sit_and_wait_grasp(self, timeout=1.0):
        if not self.is_activated(): return
        initial_fingers_position = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ])
        current_fingers_position = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ])
        current_fingers_status = np.asarray([
            self.status_raw.gDTA,
            self.status_raw.gDTB,
            self.status_raw.gDTC
        ])
        time_elapsed = 0.0
        while ( time_elapsed < timeout and \
            ( np.count_nonzero(current_fingers_status) != 3 or \
            np.sum(initial_fingers_position-current_fingers_position) == 0 )
            ):
            current_fingers_position = np.asarray([
                self.status_raw.gPOA,
                self.status_raw.gPOB,
                self.status_raw.gPOC
            ])
            current_fingers_status = np.asarray([
                self.status_raw.gDTA,
                self.status_raw.gDTB,
                self.status_raw.gDTC
            ])
            sleep( self.controller_clock_sec )
            if np.sum(initial_fingers_position-current_fingers_position) == 0:
                time_elapsed += self.controller_clock_sec


    def _sit_and_wait_activation(self, timeout=1.0):
        if self.is_activated(): return
        time_elapsed = 0.0
        while ( time_elapsed < timeout and \
            self.status_raw.gIMC != 3
            ):
            sleep( self.controller_clock_sec )
            if self.status_raw.gIMC == 3:
                time_elapsed += self.controller_clock_sec


    def _sit_and_wait_status(self, timeout=0.5):
        if self.status_raw is not None: return
        time_elapsed = 0.0
        while ( time_elapsed < timeout and \
            self.status_raw is None
            ):
            sleep( self.controller_clock_sec )
            if self.status_raw is not None:
                time_elapsed += self.controller_clock_sec


    def _sit_and_wait_scissor(self, timeout=1.0):
        # TODO: this is not used
        if not self.is_activated(): return
        initial_scissor_position = self.status_raw.gPOS
        current_scissor_position = self.status_raw.gPOS
        current_scissor_status = self.status_raw.gDTS
        time_elapsed = 0.0
        while ( time_elapsed < timeout and \
            ( current_scissor_status == 0 or \
            initial_scissor_position-current_scissor_position == 0 )
            ):
            current_scissor_position = self.status_raw.gPOS
            current_scissor_status = self.status_raw.gDTS
            sleep( self.controller_clock_sec )
            if initial_scissor_position-current_scissor_position == 0:
                time_elapsed += self.controller_clock_sec


    def _sit_and_wait_mode_switch(self, timeout=1.0):
        if self.status_raw.gACT != 1: return
        time_elapsed = 0.0
        while ( time_elapsed < timeout ):
            sleep( self.controller_clock_sec )
            if self.status_raw.gIMC == 3:
                time_elapsed += self.controller_clock_sec


    def publish_command(self, command):
        self.pub.publish(command)
        return

    def reset(self):
        self.command = SModelRobotOutput()
        self.command.rACT = 0
        self.command.rFRA = self.gripper_force
        self.publish_command(self.command)
        return self

    def get_mode(self):
        if(self.status_raw is None):
            return GraspMode.UNITIALIZED
        if(self.status_raw.gMOD == 0):
            return GraspMode.BASIC
        if(self.status_raw.gMOD == 1):
            return GraspMode.PINCH
        if(self.status_raw.gMOD == 2):
            return GraspMode.WIDE
        if(self.status_raw.gMOD == 3):
            return GraspMode.SCISSOR
        if(self.status_raw.gIMC == 2):
            return GraspMode.TRANSITIONING

    def set_mode(self, mode):
        if mode not in [GraspMode.BASIC, GraspMode.PINCH, GraspMode.WIDE, GraspMode.SCISSOR]:
            print('Commanded mode `%s` does not exist. Nothing to do!' % mode)
            return False
        modes_fcn = {
            GraspMode.BASIC : self.basic_mode,
            GraspMode.PINCH : self.pinch_mode,
            GraspMode.WIDE : self.wide_mode,
            GraspMode.SCISSOR : self.scissor_mode
        }
        mod_fcn = modes_fcn[mode]
        mod_fcn()
        return self

    def go_to(self, val):
        if not ( isinstance (val, float) or isinstance (val, int) ):
            print("Value must be of type float")
            return
        val = float(val)
        if val < 0.0:
            ROS_WARN( "Value must be in range [0,1], got %d. Using 0.0.", val )
            val = 0.0
        if val > 1.0:
            val = 1.0
        self.command.rPRA = int(val * 255.0)
        self.publish_command(self.command)
        self._sit_and_wait_grasp()
        return self

    def goto(self, val):
        return self.go_to(val)

    def is_open(self): #TODO: this does not work in scissor mode because we check the wrong axis
        epsilon = 0.08
        current_fingers_position_norm = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ]) / 255.0
        open_position_norm = np.asarray([
            self._open_value,
            self._open_value,
            self._open_value
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


    def is_closed(self): #TODO: this does not work in scissor mode because we check the wrong axis
        epsilon = 0.08
        current_fingers_position_norm = np.asarray([
            self.status_raw.gPOA,
            self.status_raw.gPOB,
            self.status_raw.gPOC
        ]) / 255.0
        closed_position_norm = np.asarray([
            self._close_value,
            self._close_value,
            self._close_value
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


    def is_activated(self):
        if self.status_raw.gACT == 1 and self.status_raw.gIMC == 3:
            return True
        else:
            return False


    def activate(self):
        self.command = SModelRobotOutput()
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rSPA = 255
        self.command.rFRA = self.gripper_force
        self.publish_command(self.command)
        self._sit_and_wait_activation()
        self.open()
        return self

    def release(self):
        self.command.rPRA = int(self._open_value * 255.0)
        self.publish_command(self.command)
        self._sit_and_wait_grasp()
        return self

    def open(self):
        return self.release()

    def grasp(self):
        self.command.rPRA = int(self._close_value * 255.0)
        self.publish_command(self.command)
        self._sit_and_wait_grasp()
        return self

    def close(self):
        return self.grasp()

    def basic_mode(self):
        self.command.rMOD = 0
        self.grasp_mode = GraspMode.TRANSITIONING
        self.publish_command(self.command)
        self._sit_and_wait_mode_switch()
        self.grasp_mode = GraspMode.BASIC
        return self

    def pinch_mode(self):
        self.command.rMOD = 1
        self.grasp_mode = GraspMode.TRANSITIONING
        self.publish_command(self.command)
        self._sit_and_wait_mode_switch()
        self.grasp_mode = GraspMode.PINCH
        return self

    def wide_mode(self):
        self.command.rMOD = 2
        self.grasp_mode = GraspMode.TRANSITIONING
        self.publish_command(self.command)
        self._sit_and_wait_mode_switch()
        self.grasp_mode = GraspMode.WIDE
        return self

    def scissor_mode(self):
        self.command.rMOD = 3
        self.grasp_mode = GraspMode.TRANSITIONING
        self.publish_command(self.command)
        self._sit_and_wait_mode_switch()
        self.grasp_mode = GraspMode.SCISSOR
        return self

    def shutdown(self, *args):
        print('Shutting down Gripper...')
        self.is_shutdown = True
        self.grasp_mode = GraspMode.UNITIALIZED
        self.controller_thread.join()
        # rospy.signal_shutdown("SIGINT signal received")
        print('Gripper released!')
        return True


def _gripper_interface_controller( gripper_obj ):
    while (not gripper_obj.is_shutdown) and (not rospy.is_shutdown()):
        time_offset = time() - gripper_obj.sub_last_status_message_t
        if time_offset > 2.0:
            print('WARN: No feedback message in the last %d seconds' % time_offset)
        sleep(2.0)
