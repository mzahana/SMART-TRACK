#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from math import radians
def generate_launch_description():
    ld = LaunchDescription()

    ns='observer'

    # Node for Drone 1
    world = {'gz_world': 'default'}
    # world = {'gz_world': 'ihunter_world'}
    model_name = {'gz_model_name': 'x500_d435'}
    autostart_id = {'px4_autostart_id': '4020'}
    instance_id = {'instance_id': '1'}
    # for 'default' world
    xpos = {'xpos': '0.0'}
    ypos = {'ypos': '0.0'}
    zpos = {'zpos': '0.1'}
    # For 'ihunter_world'
    # xpos = {'xpos': '-24.0'}
    # ypos = {'ypos': '8.0'}
    # zpos = {'zpos': '1.0'}
    headless= {'headless' : '0'}

    # PX4 SITL + Spawn x500_d435
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track'),
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_ns': ns,
            'headless': headless['headless'],
            'gz_world': world['gz_world'],
            'gz_model_name': model_name['gz_model_name'],
            'px4_autostart_id': autostart_id['px4_autostart_id'],
            'instance_id': instance_id['instance_id'],
            'xpos': xpos['xpos'],
            'ypos': ypos['ypos'],
            'zpos': zpos['zpos']
        }.items()
    )

    # MAVROS
    file_name = 'observer_px4_pluginlists.yaml'
    package_share_directory = get_package_share_directory('smart_track')
    plugins_file_path = os.path.join(package_share_directory, file_name)
    file_name = 'observer_px4_config.yaml'
    config_file_path = os.path.join(package_share_directory, file_name)
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track'),
                'mavros.launch.py'
            ])
        ]),
        launch_arguments={
            'mavros_namespace' :ns+'/mavros',
            'tgt_system': '2',
            'fcu_url': 'udp://:14541@127.0.0.1:14558',
            'pluginlists_yaml': plugins_file_path,
            'config_yaml': config_file_path,
            'base_link_frame': 'observer/base_link',
            'odom_frame': 'observer/odom',
            'map_frame': 'map',
            'use_sim_time' : 'True'

        }.items()
    )

    odom_frame = 'odom'
    base_link_frame=  'base_link'

    # Static TF map/world -> local_pose_ENU
    map_frame='map'
    map2pose_tf_node = Node(
        package='tf2_ros',
        name='map2px4_'+ns+'_tf_node',
        executable='static_transform_publisher',
        arguments=[str(xpos['xpos']), str(ypos['ypos']), str(zpos['zpos']), '0.0', '0', '0', map_frame, ns+'/'+odom_frame],
    )

    # Static TF base_link -> depth_camera
    # .15 0 .25 0 0 1.5707
    cam_x = 0.17
    cam_y = 0.0
    cam_z = 0.25
    cam_roll = radians(-90.0)
    cam_pitch = 0.0
    cam_yaw = radians(-90.0)
    cam_tf_node = Node(
        package='tf2_ros',
        name=ns+'_base2depth_tf_node',
        executable='static_transform_publisher',
        # arguments=[str(cam_x), str(cam_y), str(cam_z), str(cam_yaw), str(cam_pitch), str(cam_roll), ns+'/'+base_link['child_frame'], ns+'/depth_camera'],
        arguments=[str(cam_x), str(cam_y), str(cam_z), str(cam_yaw), str(cam_pitch), str(cam_roll), ns+'/'+base_link_frame, 'x500_d435_1/link/realsense_d435'],
        
    )

    # Transport rgb and depth images from GZ topics to ROS topics    
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        name='ros_bridge_node_depthcam',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                   '/d435/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
                   '/d435/image@sensor_msgs/msg/Image[ignition.msgs.Image',
                   '/d435/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
                   '/d435/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
                   '--ros-args', '-r', '/d435/depth_image:='+ns+'/depth_image',
                   '-r', '/d435/image:='+ns+'/image',
                   '-r', '/d435/points:='+ns+'/points',
                   '-r', '/d435/camera_info:='+ns+'/camera_info'
                   ],
    )

    # Kalman filter
    file_name = 'kf_param.yaml'
    package_share_directory = get_package_share_directory('smart_track')
    kf_file_path = os.path.join(package_share_directory, file_name)
    kf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('multi_target_kf'),
                'launch/kf_const_vel.launch.py'
            ])
        ]),
        launch_arguments={
            'detections_topic': 'yolo_detections_poses',
            'kf_ns' : '',
            'kf_yaml': kf_file_path

        }.items()
    )
    
    # YOLOv8
    yolov8_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('yolov8_bringup'),
                'launch/yolov8.launch.py'
            ])
        ]),
        launch_arguments={
            'model': '/home/user/shared_volume/ros2_ws/src/smart_track/config/drone_detection_v3.pt',
            'threshold' : '0.5',
            'input_image_topic' : 'observer/image',
            'device': 'cuda:0',
            'namespace' : ''

        }.items()
    )

    # Yolo to pose node
    yolo2pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track'),
                'yolo2pose.launch.py'
            ])
        ]),
        launch_arguments={
            'depth_topic': 'observer/depth_image',
            'debug' : 'false',
            'caminfo_topic' : 'observer/camera_info',
            'detections_poses_topic': 'yolo_detections_poses',
            'yolo_detections_topic': 'detections',
            'detector_ns' : '',
            'reference_frame' : 'observer/odom',
            'use_sim_time' : 'True'
        }.items()
    )

    # Drone marker in RViz
    quadcopter_marker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track'),
                'quadcopter_marker.launch.py'
            ])
        ]),
        launch_arguments={
            'node_ns':ns,
            'propeller_size': '0.15',                # Set propeller_size directly
            'arm_length': '0.3',                    # Set arm_length directly
            'body_color': '[0.0, 1.0, 0.0, 1.0]',   # Set body_color directly
            'propeller_color': '[1.0, 1.0, 0.0, 1.0]',  # Set propeller_color directly
            'odom_topic': '/observer/mavros/local_position/odom',     # Set odom_topic directly
        }.items(),
    )

    # Rviz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        name='sim_rviz2',
        arguments=['-d' + os.path.join(get_package_share_directory('smart_track'), 'smart_track.rviz')]
    )

    ld.add_action(gz_launch)
    ld.add_action(map2pose_tf_node)
    ld.add_action(cam_tf_node)
    ld.add_action(ros_gz_bridge)
    ld.add_action(kf_launch) # Estimates target's states based on position measurements( Reqiures yolov8_launch & yolo2pose_launch OR gt_target_tf)
    ld.add_action(yolov8_launch)
    ld.add_action(yolo2pose_launch) # Comment this if you want to use the target ground truth (gt_target_tf.launch.py)
    ld.add_action(mavros_launch)
    ld.add_action(rviz_node)
    # ld.add_action(quadcopter_marker_launch)

    return ld