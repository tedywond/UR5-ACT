#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------
#export ROS_HOSTNAME=localhost
#export ROS_MASTER_URI=http://localhost:11311

set -x
rm -rf /usr/local/lib/python3.8/dist-packages/typing.py  # There's a weird issue with typing vs Python3

# launching app
# echo "This is an empty launch script. Update it to launch your application."
# rospack find ur5-diffuser
roslaunch ur5-cartesian-controller joystick_control.launch

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
