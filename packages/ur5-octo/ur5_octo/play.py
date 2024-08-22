#! /usr/bin/env python3

from ur5_twist_reacher.reacher import UR5TwistReacher
# from ur5_twist_reacher.image_srv import CaptureImages, CaptureImage
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy
# import readchar

# Copied from twist_control.py
last_pos = Point(x=0.1, y=-0.4, z=0.4)
last_rot = Quaternion(x=0.928, y=-0.371, z=0, w=0)
delta_xy = 0.01
delta_z = 0.01
delta_rot = 0.1

KEY_TO_ACTION = {
    "w": Point(x=last_pos.x, y=last_pos.y+delta_xy, z=last_pos.z),
    "a": Point(x=last_pos.x-delta_xy, y=last_pos.y, z=last_pos.z),
    "s": Point(x=last_pos.x, y=last_pos.y-delta_xy, z=last_pos.z),
    "d": Point(x=last_pos.x+delta_xy, y=last_pos.y, z=last_pos.z),
    "q": Quaternion(x=last_rot.x, y=last_rot.y, z=last_rot.z, w=last_rot.w+delta_rot),
    "e": Quaternion(x=last_rot.x, y=last_rot.y, z=last_rot.z, w=last_rot.w-delta_rot),
    "0": Point(x=last_pos.x, y=last_pos.y, z=last_pos.z+delta_z),
    "9": Point(x=last_pos.x, y=last_pos.y, z=last_pos.z-delta_z),
}


if __name__ == '__main__':

    # Run image service
    # capture_img = CaptureImages()

    # srv_name = 'capture_images'
    # print('waiting for service...')
    # rospy.wait_for_service(srv_name)
    # print('waiting for service...done')
    # try:
    #     capture_img = rospy.ServiceProxy(srv_name, CaptureImage)
    #     img = capture_img()
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s"%e)
    # print('img', img.shape)
    # import sys; sys.exit()

    # Initializes the gripper
    twist_reacher = UR5TwistReacher()
    # twist_reacher.twist_controller._init_arm(spawn_twistvel_controller=False)

    # Copied from twist_control.py
    init_pos = Point(x=0.1, y=-0.4, z=0.4)
    init_ori = Quaternion(x=0.928, y=-0.371, z=0, w=0)

    # # receive input from keyboard
    # while True:
    #     print("Press a key: ")
    #     key = readchar.readkey()
    #     print(f"You pressed: {key}")

    #     if key in KEY_TO_ACTION:
    #         print(f"Action: {KEY_TO_ACTION[key]}")
    #         twist_reacher.twist_controller._move_to(KEY_TO_ACTION[key], last_rot, spawn_twistvel_controller=False)
    #     else:
    #         print("Invalid key")

    for i in range(10):
        pos = Point(x=0.1 + 0.02 * (i+1), y=-0.4, z=0.4)
        twist_reacher.twist_controller._move_to(pos, init_ori, spawn_twistvel_controller=False)

    # open gripper
    twist_reacher.twist_controller.grasp(0.0)

    print('execution finished!')