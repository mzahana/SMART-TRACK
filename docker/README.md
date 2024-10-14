# SMART-TRACK Docker Image

A Docker development environment for the **SMART-TRACK** framework

What is included in the Docker image is:

* **Ubuntu 22.04 (jammy)**
* **ROS 2 Humble**
* **Gazebo Garden**
* **Development tools required by PX4 firmware + PX4-Autopilot source code**
* **ROS 2 packages including SMART-TRACK simulation packags**

# Build Docker Image

Docker should be installed before proceeding with the next steps
You can follow [this link](https://docs.docker.com/engine/install/ubuntu/) for docker setup on Ubuntu

<!-- * Clone this package `git clone https://github.com/mzahana/d2dtracker_sim_docker` -->

* Build the docker image for CUDA `12.2.0` (the default in `docker_run_with_cuda.sh`)
    ```bash
    cd smart_track/docker
    make px4-simulation-cuda12.2.0-ubuntu22
    ```

* Build the docker image for CUDA `11.7.1` (NOT the default in `docker_run_with_cuda.sh`)
    ```bash
    cd smart_track/docker
    make px4-simulation-cuda11.7.1-ubuntu22
    ```

* **NOTE** If you want to build an image without `CUDA` run
    ```bash 
    cd smart_track/docker
    make px4-dev-simulation-ubuntu22
    ```

This builds a Docker image that has the required PX4-Autopilot source code and the required dependencies, ROS 2 Humble Desktop, and `ros2_ws` that contains `SMART-TRACK` and other ROS 2 packages.

The Gazebo version in the provided Docker image is Gazebo `garden`.

# Run Docker Container
* Run the following script to run and enter the docker container `smart_track_cuda` 
    ```bash
    cd smart_track
    ./docker_run_with_cuda.sh
    ```
    This will run a docker container named `smart_track_cuda` and installs `PX4-Autopilot`, and create `ros2_ws` with the required ROS 2 packages, in the shared volume.
* Once the above script is executed successfully, you should be in the docker container terminal. The docker defualt name is `smart_track_cuda`. The username inside the docker and its password is `user`

* **NOTE** If you built the docker image with no `CUDA`, run the container using
    ```bash
    cd smart_track
    ./docker_run_no_cuda.sh
    ```
    
    The container name in this case is `smart_track`

# Build ros2_ws

Enter the docker container, and execute the following

```bash
cd ~/shared_volume
# Execute the following step, if you have not created the ros2_ws already
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/mzahana/SMART-TRACK.git smart_track
cd smart_track
./install.sh
```

# PX4 Autopilot & ROS 2 Wrokspace

* You can find the `ros2_ws` and the `PX4-Autopilot` inside the shared volume. The path of shared volume inside the container is `/home/user/shared_volume`. The path of the shared volume outside the container is `$HOME/smart_track_shared_volume` (or `$HOME/smart_track_cuda_shared_volume` if you built the image with cuda).

# Run Simulation

* Follow the instructions of the [SMART-TRACK](https://github.com/mzahana/SMART-TRACK#run) package to run the simulation.
* You can re-enter the docker container by executing the  `docker_run_with_cuda.sh`

# Troublshooting

## QGC not running when using the container in WSL2
This is likely due to the network isolation between Windows/WSL2/Docker.

* Setup your `.wslconfig` file properly to mirror system network. Here is my `.wslconfig` (should be in your Windows Home folder)
```text
# Settings apply across all Linux distros running on WSL 2
[wsl2]
firewall=false
networkingMode=mirrored
# Limits VM memory to use no more than 4 GB, this can be set as whole numbers using GB or MB
memory=80% 

# Sets the VM to use two virtual processors
processors=24

# Sets amount of swap storage space to 8GB, default is 25% of available RAM
swap=16GB

# Sets swapfile path location, default is %USERPROFILE%\AppData\Local\Temp\swap.vhdx
#swapfile=C:\\temp\\wsl-swap.vhdx

# Disable page reporting so WSL retains all allocated memory claimed from Windows and releases none back when free
#pageReporting=false

# Turn on default connection to bind WSL 2 localhost to Windows localhost
#localhostforwarding=true

# Disables nested virtualization
nestedVirtualization=false

# Turns on output console showing contents of dmesg when opening a WSL 2 distro for debugging
debugConsole=false

# Enable experimental features
[experimental]
hostAddressLoopback=true
#sparseVhd=true
```
* Open Dopcker Desktop application, go to settings (press gear icon on top right). Go to `Resource -> Network ` tab and enble `Enable host networking`. You will need to sign-in into your Docker account to enable this option.

* Restart your docker desktop application

* Stop and delete the d2dtracker container. Then, run it again.
* Now, you should be able to use QGroundControl installed in Windows to monitor and control PX4 SITL running inside the container.

**Another method to run QGC inside the container**

If you are running the simulation container inside WSL2 and want to use QGroundControl inside the container, it's recommended to [download](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html#ubuntu) the `.Appimage` of QGC inside the shared volume. Then, inside the container, run
```bash
./QGroundControl.AppImage --appimage-extract
```
This command creates a directory named `squashfs-root` containing all the files.
The run you can run QGC
```bash
cd squashfs-root
./AppRun
```