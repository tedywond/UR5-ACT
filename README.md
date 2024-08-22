# Run inference
0. Change the arguments of the run function in `/home/ripl/workspace/ripl/ur5-octo/packages/ur5-octo/ur5_octo/inference.py`.


1.  Set up the ur5 arm and the two cameras following https://github.com/ripl/docs/wiki/UR5-Arm

2. Open a new terminal, go to `/home/ripl/workspace/ripl/ur5-octo` and run

`cpk run --name server -f -M -L bash --net=host`

This will launch a cpk container, which is a thin wrapper around docker, that has all the ros dependencies installed. If a container of the same name is already running you can just kill it by running `docker kill [container id]`. You can see what the arguments mean by running `cpk run -h`.

4. Then, inside the containerm you can launch the twist control server by running

`rosrun ur5_twist_control twist_control_server.py`.

The twist control server is used to host the rosservice `gohomepose` which upon calling will reset the arm's pose.

4. Open a new terminal, go to `/home/ripl/workspace/ripl/ur5-octo` and run

`cpk run --name octo -f -M -L bash --net=host -- --gpus all`

5. In the container, can launch the inference script by running

`rosrun ur5-octo inference.py`.

It takes around a minute to load the model.

6. If some error pop up saying it cannot load the model weights, try running

`sudo chown -R root:root *`

in the folder where the weights are stored to change the owner of the directory.
# UR5-ACT
