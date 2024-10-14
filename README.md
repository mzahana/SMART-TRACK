# SMART-TRACK

A novel Kalman Filter-Guided Sensor Fusion For Robust Robot Object Tracking in Dynamic Environments.

**SMART-TRACK** is a ROS2-based framework designed for real-time, precise detection and tracking of multiple objects in dynamic environments. By combining advanced object detection techniques with Kalman Filter estimators, SMART-TRACK maintains robust tracking continuity even when direct measurements are intermittent or fail.

## Table of Contents

- [SMART-TRACK](#smart-track)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Framework Architecture](#framework-architecture)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Getting Started](#getting-started)
  - [Subscribed and Published Topics](#subscribed-and-published-topics)
    - [Subscribed Topics](#subscribed-topics)
    - [Published Topics](#published-topics)
  - [Customization](#customization)
  - [Contributing](#contributing)
  - [Notes](#notes)

## Overview

In sensor fusion and state estimation for object detection and localization, the Kalman Filter (KF) is a standard framework. However, its effectiveness diminishes when measurements are not continuous, leading to rapid divergence in state estimations. **SMART-TRACK** introduces a novel approach that leverages high-frequency state estimates from the Kalman Filter to guide the search for new measurements. This method maintains tracking continuity even when direct measurements falter, making it pivotal in dynamic environments where traditional methods struggle. While applicable to various sensor types, our demonstration focuses on depth cameras within a ROS2-based simulation environment.

![alt text](<images/SMART-TRACK (5).png>)

## Key Features

- **Robust Multi-Target Tracking**: Maintains continuous tracking even when object detections are intermittent or fail.
- **Measurement Augmentation**: Uses Kalman Filter feedback to augment measurements, enhancing detection reliability.
- **Versatile Sensor Integration**: Adaptable to various sensor types (e.g., LiDAR, depth cameras) by transforming KF predictions into the sensor's frame.
- **ROS2 Compatibility**: Fully compatible with ROS2, facilitating integration into modern robotic systems.
- **Open-Source Implementation**: Provides an open-source ROS2-compatible implementation for broader adaptation and further advancements.

## Framework Architecture

![alt text](images/system_architecture.png)

This diagram illustrates the **SMART-TRACK** workflow for robust UAV object tracking using RGB and depth images. Here's a simplified explanation:

1. **RGB Image ( \( I_{RGB} \) )**: The system uses **YOLOv8** to detect the object and generate a bounding box. If detected, the 3D position is computed and sent to the **Kalman Filter (KF)**.

2. **Depth Image ( \( I_{depth} \) )**: Simultaneously, depth images are used to compute the object's 3D position, which is fed as a measurement to the **Kalman Filter (KF)**.

3. **Kalman Filter (KF)**: The KF predicts the object's position and updates it using new measurements. When **YOLOv8** fails to detect the object, the KF’s prediction is used to guide a **search region (ROI)** in the RGB image.

4. **Tracking**: The system keeps tracking the object even if direct detection fails, using the Kalman Filter to estimate and predict the object's location.

## Installation

### Prerequisites

- **ROS2 Humble**
- **Python 3.x**
- **OpenCV**
- **PX4 v1.14**
- **TensorFlow** or **PyTorch** (depending on the object detection model used)
- **Additional Dependencies**:
  - `vision_msgs`
  - `rclpy`
  - `cv_bridge`

### Setup

1. **Clone the SMART-TRACK Repository**

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/mzahana/SMART-TRACK smart_track
   ```

2. **Install YOLOv8**

   SMART-TRACK uses YOLOv8 for object detection. Ensure YOLOv8 is installed before proceeding.

   - Clone [yolov8_ros](https://github.com/mgonzs13/yolov8_ros/releases/tag/2.0.1) and use release `2.0.1`, which we tested with YOLOv8 at commit `b638c4ed`.

     ```bash
     cd ~/ros2_ws/src
     git clone -b 2.0.1 https://github.com/mgonzs13/yolov8_ros.git
     pip3 install -r yolov8_ros/requirements.txt
     cd ~/ros2_ws
     rosdep install --from-paths src --ignore-src -r -y
     colcon build
     ```

3. **Download the Custom YOLOv8 Model**

   - A custom YOLOv8 model for drone detection is available in the [`config`](config) directory of this package. The model is named `drone_detection_v3.pt`. Use this model with the `yolov8_ros` package.

4. **Clone the Kalman Filter Implementation**

   - Clone the [multi_target_kf](https://github.com/mzahana/multi_target_kf/tree/ros2_humble) repository into your `ros2_ws/src` directory. Checkout the `ros2_humble` branch.

     ```bash
     git clone -b ros2_humble https://github.com/mzahana/multi_target_kf.git
     ```

5. **Build the ROS2 Workspace**

   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

## Getting Started

To run the pose estimator, follow these steps:

1. **Ensure All Packages Are Built and Sourced**

   - Make sure that all packages are inside the ROS2 workspace.
   - Build the workspace if you haven't already:

     ```bash
     cd ~/ros2_ws
     colcon build
     source install/setup.bash
     ```

2. **Launch the Pose Estimator**

   ```bash
   ros2 launch smart_track yolo2pose.launch.py
   ```

   - This launch file starts the pose estimation node that processes YOLO detections to estimate the poses of detected objects.

3. **Optional: Run the Kalman Filter Tracker**

   - The `yolo2pose_node.py` accepts Kalman Filter estimations of the target's 3D position to implement the KF-guided measurement algorithm for more robust state estimation.
   - Launch the Kalman Filter tracker:

     ```bash
     ros2 launch multi_target_kf multi_target_kf.launch.py
     ```

## Subscribed and Published Topics

### Subscribed Topics

- **Image and Camera Information**

  - `/interceptor/depth_image` (`sensor_msgs/Image`): Depth image stream from the camera sensor.
  - `/interceptor/camera_info` (`sensor_msgs/CameraInfo`): Camera calibration and configuration data.

- **Object Detections**

  - `/detections` (`vision_msgs/DetectionArray`): Detection results containing bounding boxes, classes, and confidence scores.

- **Kalman Filter Tracks**

  - `/kf/good_tracks` (`kf_msgs/KFTracks`): Filtered and predicted states of the tracked objects from the Kalman Filter.

### Published Topics

- **Pose Arrays**

  - `/yolo_poses` (`geometry_msgs/PoseArray`): Array of poses estimated from YOLO detections.

- **Overlay Images**

  - `/overlay_yolo_image` (`sensor_msgs/Image`): Image with overlaid detections and tracking information for visualization.

## Customization

To adapt SMART-TRACK for different types of objects or sensors:

- **Modify the Detection Model**: Replace or retrain the object detection model (e.g., YOLOv8) to detect your objects of interest.
- **Adjust Kalman Filter Parameters**: Tweak the parameters in `multi_target_kf` for optimal tracking performance.
- **Integrate Different Sensors**: Adapt the measurement augmentation system to work with other sensors by transforming KF predictions into the appropriate sensor frame.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   - Click the 'Fork' button at the top right of this page.

2. **Clone Your Fork**

   - Replace `your-username` with your GitHub username.

     ```bash
     git clone https://github.com/your-username/smart_track.git
     ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   - Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -am 'Add new feature'
   ```

6. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**

   - Go to the original repository and click `New Pull Request`.

## Notes

- **Depth Image and Camera Info Topics**: Ensure you provide the correct depth image topic and camera info topic in the [`detection.launch.py`](launch/detection.launch.py) file.
- **Static Transformation**: There should be a valid static transformation between the robot's base link frame and the camera frame. This is required to compute the position of the detected objects in the observer's localization frame, which can be sent to the Kalman Filter. See an example [here](https://github.com/mzahana/d2dtracker_sim/blob/5ea454e95fd292ab16cb3d28c50bb2182572ad52/launch/interceptor.launch.py#L94).
- **Configuration Parameters**: You can configure the depth-based detection parameters in the [`detection_param.yaml`](config/detection_param.yaml) file.
- **Rebuild Workspace After Modifications**: After any modifications, rebuild your workspace using:

  ```bash
  colcon build
  ```
