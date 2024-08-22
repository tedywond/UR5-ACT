#! /usr/bin/env python3

from tkinter import W
import rospy
import signal
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from threading import BoundedSemaphore
import numpy as np
from ros_utils import ROS_INFO, ROS_WARN
from capture_images.srv import CaptureImage, CaptureImageRequest
import functools


class CaptureImages:
    def __init__(self) -> None:
        # get parameters
        rospy.init_node('capture_images')


        # NOTE: No idea why ~joy_topic is not found.
        self.arm_rgb_topic = rospy.get_param('~arm_rgb_topic', '/camera_arm/color/image_raw')
        self.arm_rgb_buffer = None

        self.arm_depth_topic = rospy.get_param('~arm_depth_topic', '/camera_arm/aligned_depth_to_color/image_raw')
        self.arm_depth_buffer = None

        self.arm_pc_topic = rospy.get_param('~arm_pc', '/camera_arm/depth_registered/points')
        self.arm_pc_buffer = None

        # Stores image frame
        self._cam2buffer = {
            'arm_rgb': self.arm_rgb_buffer,
            'arm_depth': self.arm_depth_buffer,
            'arm_pc': self.arm_pc_buffer
        }

        # Binary flag that specifies whether to record the buffer
        self._cam2capture = {cam: False for cam in self._cam2buffer.keys()}

        ROS_INFO("Subscribing to image topics...")
        # subscribe to joy commands
        # NOTE: If buffer size is small, it will take longer to receive the whole image (?)
        self._sub_arm_rgb = rospy.Subscriber(self.arm_rgb_topic, Image, functools.partial(self.img_cb, 'arm_rgb'), queue_size=1, buff_size=65535 * 100)
        self._sub_arm_depth = rospy.Subscriber(self.arm_depth_topic, Image, functools.partial(self.img_cb, 'arm_depth'), queue_size=1, buff_size=65535 * 100)
        self._sub_arm_pc = rospy.Subscriber(self.arm_pc_topic, PointCloud2, functools.partial(self.img_cb, 'arm_pc'), queue_size=1, buff_size=65535 * 100)


        ROS_INFO("hello!! Running the service now...")
        self._srv = rospy.Service('capture_image', CaptureImage, self.get_image)

    def img_cb(self, cam, frame):
        """ Stores the frame only if `capture` is True.
        """
        if cam not in self._cam2buffer:
            raise KeyError(f'Unknown camera {cam}')

        if self._cam2capture[cam]:
            self._cam2buffer[cam] = frame
        else:
            self._cam2buffer[cam] = None

    def get_image(self, request: CaptureImageRequest):
        ROS_INFO(f'Request received!!: {request}')
        cam = request.camera.data
        if cam not in self._cam2buffer:
            raise KeyError(f'Unknown camera {cam}')

        # Set capture flag to True
        import time
        now = time.perf_counter()
        self._cam2capture[cam] = True
        while self._cam2buffer[cam] is None:
            rospy.sleep(0.02)
        elapsed = time.perf_counter() - now
        ROS_INFO(f'Took {elapsed:.3f} seconds to get the image!')

        img = self._cam2buffer[cam]
        
        # Reset the flags
        self._cam2capture = {cam: False for cam in self._cam2buffer.keys()}
        ROS_INFO(f'Request processed!!')

        return img


if __name__ == '__main__':
    ROS_INFO('THis is running')
    image_capturer = CaptureImages()
    ROS_INFO('CaptureImages is instantiated')

    # keep spinning
    rospy.spin()
