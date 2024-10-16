#!/bin/bash -e

# This script sets up the D2DTracker simulation environment

if [ -z "${DEV_DIR}" ]; then
  echo "Error: DEV_DIR environment variable is not set. Set it using export DEV_DIR=<DEV_DIR_deirectory_that_should_contain_PX4-Autopilot_and_ros2_ws>"
  exit 1
fi
echo "DEV_DIR=$DEV_DIR"
sleep 1
echo "GIT_USER=$GIT_USER"
echo "GIT_TOKEN=$GIT_TOKEN"
sleep 1

ROS2_WS=$DEV_DIR/ros2_ws
ROS2_SRC=$DEV_DIR/ros2_ws/src
PX4_DIR=$DEV_DIR/PX4-Autopilot
OSQP_SRC=$DEV_DIR

# # Make sure that PX4 root directory is set
# if [ -z "${PX4_ROOT}" ]; then
#   echo "Error: PX4_ROOT environment variable is not set. Set it using export PX4_ROOT=<PX4-ROOT_deirectory_that_contains_PX4-Autopilot>"
#   exit 1
# fi

# # Make sure that ROS2_WS directory is set
# if [ -z "${ROS2_WS}" ]; then
#   echo "Error: ROS2_WS environment variable is not set. Set it using export ROS2_WS=<ROS2_WS_deirectory_that_contains_ros2_ws>"
#   exit 1
# fi

if [ ! -d "$ROS2_WS" ]; then
  echo "Creating $ROS2_SRC"
  mkdir -p $ROS2_SRC
fi

SIM_PKG_URL=''
if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
    SIM_PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/SMART-TRACK.git
else
    SIM_PKG_URL=https://github.com/mzahana/SMART-TRACK.git
fi

# Clone the d2dtracker_sim if it doesn't exist
if [ ! -d "$ROS2_SRC/smart_track" ]; then
    cd $ROS2_SRC
    git clone $SIM_PKG_URL smart_track && cd $ROS2_SRC/smart_track && git pull origin main
else
    cd $ROS2_SRC/smart_track && git pull origin main
fi

# Clone and build PX4-Autopilot if it doesn't exist
if [ ! -d "$PX4_DIR" ]; then
    echo "Cloning $PX4_DIR..."
    cd $DEV_DIR
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive
    cd $PX4_DIR
    git checkout v1.14.0
    make submodulesclean
    make clean
    make distclean
else
    echo "PX4_DIR=$PX4_DIR already exists"
    cd $PX4_DIR
    make submodulesclean
    make clean
    make distclean
    git checkout v1.14.0
    make submodulesclean
    make clean
    make distclean
fi

# Build px4_sitl
cd $PX4_DIR && make px4_sitl

# Copy files to $PX4_DIR
echo && echo  "Copying files to ${PX4_DIR}" && echo
sleep 1
cp -r ${ROS2_SRC}/smart_track/models/* ${PX4_DIR}/Tools/simulation/gz/models/
cp -r ${ROS2_SRC}/smart_track/worlds/* ${PX4_DIR}/Tools/simulation/gz/worlds/
cp -r ${ROS2_SRC}/smart_track//config/px4/* ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/
cd $PX4_DIR && make px4_sitl

# Clone some PX4 rose-related packages
# px4_msgs not needed anymore since we are using mavros, but kkeeping it in case it is needed in future
# if [ ! -d "$ROS2_SRC/px4_msgs" ]; then
#     cd $ROS2_SRC
#     git clone https://github.com/PX4/px4_msgs.git
# else
#     cd $ROS2_SRC/px4_msgs && git pull origin main
# fi

#
# custom px4_ros_com
# Not needed anymore since we are using mavros, but kkeeping it in case it is needed in future
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     echo "GIT_USER=$GIT_USER , GIT_TOKEN=$GIT_TOKEN"
#     echo
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/px4_ros_com.git
# else
#     echo "GIT_USER and GIT_TOKEN are not set"
#     PKG_URL=https://github.com/mzahana/px4_ros_com.git
# fi

# if [ ! -d "$ROS2_SRC/px4_ros_com" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL}
# else
#     cd $ROS2_SRC/px4_ros_com && git checkout main && git pull origin main
# fi



#
# d2dtracker_drone_detector
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/d2dtracker_drone_detector.git
# else
#     PKG_URL=https://github.com/mzahana/d2dtracker_drone_detector.git
# fi

# if [ ! -d "$ROS2_SRC/d2dtracker_drone_detector" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL}
#     cd $ROS2_SRC/d2dtracker_drone_detector && git checkout ros2_humble && git pull origin ros2_humble
# else
#     cd $ROS2_SRC/d2dtracker_drone_detector && git checkout ros2_humble && git pull origin ros2_humble
# fi


#
# multi_target_kf
#
PKG_URL=''
if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
    PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/multi_target_kf.git
else
    PKG_URL=https://github.com/mzahana/multi_target_kf.git
fi

if [ ! -d "$ROS2_SRC/multi_target_kf" ]; then
    cd $ROS2_SRC
    git clone ${PKG_URL} -b ros2_humble
else
    cd $ROS2_SRC/multi_target_kf && git checkout ros2_humble && git pull origin ros2_humble
fi

#
# custom_trajectory_msgs
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/custom_trajectory_msgs.git
# else
#     PKG_URL=https://github.com/mzahana/custom_trajectory_msgs.git
# fi

# if [ ! -d "$ROS2_SRC/custom_trajectory_msgs" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL} -b ros2_humble
# else
#     cd $ROS2_SRC/custom_trajectory_msgs && git checkout ros2_humble && git pull origin ros2_humble
# fi

#
# trajectory_prediction
# Constant celocity and Bezier based trajectory prediction
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     echo "GIT_USER=$GIT_USER , GIT_TOKEN=$GIT_TOKEN"
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/trajectory_prediction.git
# else
#     echo "GIT_USER and GIT_TOKEN are not set"
#     PKG_URL=https://github.com/mzahana/trajectory_prediction.git
# fi

# if [ ! -d "$ROS2_SRC/trajectory_prediction" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL} -b ros2_humble
# else
#     cd $ROS2_SRC/trajectory_prediction && git checkout ros2_humble && git pull origin ros2_humble
# fi
# cd $ROS2_SRC/trajectory_prediction && . setup.sh

#
# drone_path_predictor_ros
# GRU-based trajectory prediction
# This is better than trajectory_prediction
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     echo "GIT_USER=$GIT_USER , GIT_TOKEN=$GIT_TOKEN"
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/drone_path_predictor_ros.git
# else
#     echo "GIT_USER and GIT_TOKEN are not set"
#     PKG_URL=https://github.com/mzahana/drone_path_predictor_ros.git
# fi

# if [ ! -d "$ROS2_SRC/drone_path_predictor_ros" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL} -b ros2_humble
# else
#     cd $ROS2_SRC/drone_path_predictor_ros && git checkout ros2_humble && git pull origin ros2_humble
# fi

#
# trajectory_generation
# MPC-based trajectory generation
#
# PKG_URL=''
# if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
#     echo "GIT_USER=$GIT_USER , GIT_TOKEN=$GIT_TOKEN"
#     PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/trajectory_generation.git
# else
#     echo "GIT_USER and GIT_TOKEN are not set"
#     PKG_URL=https://github.com/mzahana/trajectory_generation.git
# fi

# if [ ! -d "$ROS2_SRC/trajectory_generation" ]; then
#     cd $ROS2_SRC
#     git clone ${PKG_URL} -b ros2_humble
# else
#     cd $ROS2_SRC/trajectory_generation && git checkout ros2_humble && git pull origin ros2_humble
# fi
# cd $ROS2_SRC/trajectory_generation && . setup.sh

#
# custom mav_controllers_ros
#
PKG_URL=''
if [[ -n "$GIT_USER" ]] && [[ -n "$GIT_TOKEN" ]]; then
    echo "GIT_USER=$GIT_USER , GIT_TOKEN=$GIT_TOKEN"
    echo
    PKG_URL=https://$GIT_USER:$GIT_TOKEN@github.com/mzahana/mav_controllers_ros.git
else
    echo "GIT_USER and GIT_TOKEN are not set"
    PKG_URL=https://github.com/mzahana/mav_controllers_ros.git
fi

if [ ! -d "$ROS2_SRC/mav_controllers_ros" ]; then
    cd $ROS2_SRC
    git clone ${PKG_URL}
    cd $ROS2_SRC/mav_controllers_ros && git checkout ros2_humble
else
    cd $ROS2_SRC/mav_controllers_ros && git checkout ros2_humble && git pull origin ros2_humble
fi

#
# yolov8
#
if [ ! -d "$ROS2_SRC/yolov8_ros" ]; then
    cd $ROS2_SRC
    git clone https://github.com/mgonzs13/yolov8_ros.git
    cd $ROS2_SRC/yolov8_ros && git checkout 2.0.1
else
    cd $ROS2_SRC/yolov8_ros && git pull origin && git checkout 2.0.1
fi

#
# MAVROS
#
# these mavlink and mavros versions are working for ros2 humble
# Sept 17, 2023
echo "Cloning mavlink package ... " && sleep 1
if [ ! -d "$ROS2_SRC/mavlink" ]; then
    cd $ROS2_SRC
    git clone  https://github.com/ros2-gbp/mavlink-gbp-release.git mavlink
    cd $ROS2_SRC/mavlink && git checkout release/humble/mavlink/2023.9.9-1
fi
# Custom mavros pkg is required to handle TF issues in multi-vehicle simulation
echo "Cloning custom mavros package ... " && sleep 1
if [ ! -d "$ROS2_SRC/mavros" ]; then
    cd $ROS2_SRC
    git clone  https://github.com/mzahana/mavros.git
    cd $ROS2_SRC/mavros && git checkout ros2_humble
fi

cd $ROS2_WS && rosdep init && rosdep update && rosdep install --from-paths src --ignore-src -r -y

cd $ROS2_WS && MAKEFLAGS='j1 -l1' colcon  build --packages-up-to mavros --executor sequential

cd $ROS2_WS && MAKEFLAGS='j1 -l1' colcon build --packages-up-to mavros_extras --executor sequential

cd $ROS2_WS && colcon build

echo "DONE. Pkgs are built. Models and airframe config files are copied to the respective folder in the ${PX4_DIR} directory"
echo "Source the ros2_ws and use <ros2 launch smart_track observer.launch.py> to run the simulation"
cd $HOME