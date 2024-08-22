#! /usr/bin/env python3

from ur5_twist_reacher.reacher import UR5TwistReacher
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy
import json

def data_to_pose(data):
    pos = Point(x=data["translation"][0], y=data["translation"][1], z=data["translation"][2])
    rot = Quaternion(x=data["rotation"][0], y=data["rotation"][1], z=data["rotation"][2], w=data["rotation"][3])
    return pos, rot




if __name__ == "__main__":
    twist_reacher = UR5TwistReacher()
    twist_reacher.twist_controller._init_arm(spawn_twistvel_controller=False)

    with open("test.json", 'r') as file:
        traj_list = json.load(file)

    init_pos, init_ori = data_to_pose(traj_list[0])
    twist_reacher.twist_controller._move_to(init_pos, init_ori, spawn_twistvel_controller=False)

    for data in traj_list:
        pos, rot = data_to_pose(data)
        print(f"Moving to {pos}, {rot}")
        twist_reacher.twist_controller._add_goal(pos, rot, freq=5)
        # twist_reacher.twist_controller._move_to(pos, init_ori, spawn_twistvel_controller=False)
        twist_reacher.twist_controller.grasp(data["gripper_state"])

        # save image to corresponding folder


    print('execution finished!')

    # rosrun ur5_octo replay.py
