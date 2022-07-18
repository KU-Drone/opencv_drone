#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# cd ../ardupilot/ArduCopter && sim_vehicle.py -v ArduCopter -f gazebo-iris --console
# cd ../ardupilot/ArduCopter
cd $SCRIPT_DIR/../ArduCopter
pwd
sim_vehicle.py -v ArduCopter -f gazebo-iris --console
# x=1
# while [$x -e 1]
# do
# done    