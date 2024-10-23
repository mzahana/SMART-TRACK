#! /bin/bash

# This is needed to avoid permission issues
# Ref: https://github.com/microsoft/WSL/issues/7507#issuecomment-1564150300
sudo chmod 666 /dev/dri/*
sudo chmod 666 /dev/dxg

# Use Nvidia GPU
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

DOCKER_REPO="mzahana/px4.simulation.ubuntu22.wsl"
CONTAINER_NAME="smart_track_wsl"
WORKSPACE_DIR=~/${CONTAINER_NAME}_shared_volume
CMD=""



DOCKER_OPTS=""
DOCKER_OPTS="-v /tmp/.X11-unix:/tmp/.X11-unix"
DOCKER_OPTS="$DOCKER_OPTS -v /mnt/wslg:/mnt/wslg"
DOCKER_OPTS="$DOCKER_OPTS -e DISPLAY=$DISPLAY"
DOCKER_OPTS="$DOCKER_OPTS -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
# DOCKER_OPTS="$DOCKER_OPTS -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
# DOCKER_OPTS="$DOCKER_OPTS -e PULSE_SERVER=$PULSE_SERVER"
# access to the vGPU
DOCKER_OPTS="$DOCKER_OPTS  -v /usr/lib/wsl:/usr/lib/wsl"
DOCKER_OPTS="$DOCKER_OPTS --device=/dev/dxg"
DOCKER_OPTS="$DOCKER_OPTS -e LD_LIBRARY_PATH=/usr/lib/wsl/lib"
# access to vGPU accelerated video
DOCKER_OPTS="$DOCKER_OPTS -e LIBVA_DRIVER_NAME=d3d12"
DOCKER_OPTS="$DOCKER_OPTS --device /dev/dri/card0"
DOCKER_OPTS="$DOCKER_OPTS --device /dev/dri:/dev/dri"
# DOCKER_OPTS="$DOCKER_OPTS --device /dev/dri/renderD128"
# https://marinerobotics.gtorg.gatech.edu/running-ros-with-gui-in-docker-using-windows-subsystem-for-linux-2-wsl2/
# DOCKER_OPTS="$DOCKER_OPTS -e DISPLAY=host.docker.internal:0.0"
# DOCKER_OPTS="$DOCKER_OPTS -e LIBGL_ALWAYS_INDIRECT=0"
DOCKER_OPTS="$DOCKER_OPTS -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA"
# DOCKER_OPTS="$DOCKER_OPTS -p 14550:14550"
DOCKER_OPTS="$DOCKER_OPTS --gpus all"
DOCKER_OPTS="$DOCKER_OPTS -e NVIDIA_DRIVER_CAPABILITIES=all"
echo "GPU arguments: $DOCKER_OPTS"

SUDO_PASSWORD="user"

# This will enable running containers with different names
# It will create a local workspace and link it to the image's catkin_ws
if [ "$1" != "" ]; then
    CONTAINER_NAME=$1
fi
WORKSPACE_DIR=~/${CONTAINER_NAME}_shared_volume
if [ ! -d $WORKSPACE_DIR ]; then
    mkdir -p $WORKSPACE_DIR
fi
echo "Container name:$CONTAINER_NAME WORSPACE DIR:$WORKSPACE_DIR" 


echo "Shared WORKSPACE_DIR: $WORKSPACE_DIR";

 
echo "Starting Container: ${CONTAINER_NAME} with REPO: $DOCKER_REPO"

CMD="export DEV_DIR=/home/user/shared_volume && \
    export PX4_DIR=\$DEV_DIR/PX4-Autopilot &&\
    export ROS2_WS=\$DEV_DIR/ros2_ws &&\
    export OSQP_SRC=\$DEV_DIR &&\
        source /home/user/.bashrc &&\
        if [ -f "/home/user/shared_volume/ros2_ws/install/setup.bash" ]; then
            source /home/user/shared_volume/ros2_ws/install/setup.bash
        fi &&\
         /bin/bash"
if [[ -n "$GIT_TOKEN" ]] && [[ -n "$GIT_USER" ]]; then
    CMD="export GIT_USER=$GIT_USER && export GIT_TOKEN=$GIT_TOKEN && $CMD"
fi

if [[ -n "$SUDO_PASSWORD" ]]; then
    CMD="export SUDO_PASSWORD=$SUDO_PASSWORD && $CMD"
fi

# echo $CMD

if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
        # cleanup
        echo "Restarting the container..."
        docker start ${CONTAINER_NAME}
    fi

    docker exec --user user -it ${CONTAINER_NAME} env TERM=xterm-256color bash -c "${CMD}"

else

    CMD="export DEV_DIR=/home/user/shared_volume &&\
        export PX4_DIR=\$DEV_DIR/PX4-Autopilot &&\
        export ROS2_WS=\$DEV_DIR/ros2_ws &&\
        export OSQP_SRC=\$DEV_DIR &&\
        source /home/user/.bashrc &&\
        if [ -f "/home/user/shared_volume/ros2_ws/install/setup.bash" ]; then
            source /home/user/shared_volume/ros2_ws/install/setup.bash
        fi &&\
        /bin/bash"

    if [[ -n "$GIT_TOKEN" ]] && [[ -n "$GIT_USER" ]]; then
    CMD="export GIT_USER=$GIT_USER && export GIT_TOKEN=$GIT_TOKEN && $CMD"
    fi

    if [[ -n "$SUDO_PASSWORD" ]]; then
        CMD="export SUDO_PASSWORD=$SUDO_PASSWORD && $CMD"
    fi

    echo "Running container ${CONTAINER_NAME}..."

    docker run -it \
        --network host \
        -e LOCAL_USER_ID="$(id -u)" \
        -e FASTRTPS_DEFAULT_PROFILES_FILE=/usr/local/share/middleware_profiles/rtps_udp_profile.xml \
        --volume="$WORKSPACE_DIR:/home/user/shared_volume:rw" \
        --name=${CONTAINER_NAME} \
        --privileged \
        --workdir /home/user/shared_volume \
        $DOCKER_OPTS \
        ${DOCKER_REPO} \
        bash -c "${CMD}"
fi

