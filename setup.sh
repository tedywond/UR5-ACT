#!/bin/bash

# NOTE: this setup script will be executed right before the launcher file inside the container,
#       use it to configure your environment.

set -e

source /code/devel/setup.bash


# Running it on base1 (husky-ur5)
export ROS_MASTER_URI=http://base1.local:11311
export ROS_HOSTNAME=$(hostname).local
# export ROS_MASTER_URI=http://192.168.131.11:11311
# export ROS_HOSTNAME=192.168.131.245

# Running it on oct
# unset ROS_HOSTNAME
# export ROS_MASTER_URI=http://192.168.1.1:11311
# export ROS_IP=192.168.1.1
