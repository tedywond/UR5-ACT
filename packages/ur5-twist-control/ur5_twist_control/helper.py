from geometry_msgs.msg import Point, Quaternion
import numpy as np


def get_collision_table(planning_frame):
    import rospy
    from geometry_msgs.msg import Pose, PoseStamped
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    # Specify table dimensions
    table_position = [0.9, 0.0, 0.1]
    # table_size = [0.7, 1.0, 0.48]
    table_size = [0.5, 0.5, 0.45]

    # add the table to the scene
    table = CollisionObject()
    table.header.frame_id = planning_frame
    table.header.stamp = rospy.Time.now()
    table.id = "checkers_table_volume"
    table_shape = SolidPrimitive()
    table_shape.type = SolidPrimitive.BOX
    table_shape.dimensions = table_size
    table_pose = Pose()
    table_pose.position.x = table_position[0]
    table_pose.position.y = table_position[1]
    table_pose.position.z = table_position[2]
    table_pose.orientation.w = 1.0
    table.primitives.append(table_shape)
    table.primitive_poses.append(table_pose)
    table.operation = CollisionObject.ADD
    return table


def get_arm():
    from ur5_interface import Arm, LinkType

    # Add finger tip (Robotiq)
    links = {
        'finger_tip': (LinkType.TERMINAL, 'finger_tip_manipulator')
    }

    # create arm interface
    arm = Arm(links=links)
    # set speed of the arm
    arm.finger_tip._group.set_max_velocity_scaling_factor(0.1)
    arm.finger_tip._group.set_max_acceleration_scaling_factor(0.01)

    # Newly added constraints
    arm.finger_tip._group.set_goal_position_tolerance(0.02)
    arm.finger_tip._group.set_goal_orientation_tolerance(0.3)
    arm.finger_tip._group.allow_replanning(True)
    arm.finger_tip._group.set_num_planning_attempts(10)  # Default: 1
    return arm


def get_planning_scene(arm):
    from moveit_commander import PlanningSceneInterface
    _scene = PlanningSceneInterface()
    planning_frame = arm.finger_tip.get_planning_frame()
    table = get_collision_table(planning_frame)  # Need to expose the table??
    return _scene, table


def point2numpy(point: Point):
    import numpy as np
    return np.array([point.x, point.y, point.z])
vec32numpy = point2numpy  # shortcut


def ori2numpy(ori: Quaternion):
    import numpy as np
    return np.array([ori.x, ori.y, ori.z, ori.w])


def numpy2quat(quat: np.ndarray):
    x, y, z, w = quat.tolist()
    return Quaternion(x=x, y=y, z=z, w=w)
