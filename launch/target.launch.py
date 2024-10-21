#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    ld = LaunchDescription()

    # Node for Drone 2
    model_name = {'gz_model_name': 'x3_uav'}
    autostart_id = {'px4_autostart_id': '4021'}
    instance_id = {'instance_id': '2'}
    # For default world
    xpos = {'xpos': '4.0'}
    ypos = {'ypos': '0.0'}
    zpos = {'zpos': '0.1'}
    # For ihunter_world
    # xpos = {'xpos': '200.0'}
    # ypos = {'ypos': '100.0'}
    # zpos = {'zpos': '7.0'}
    headless= {'headless' : '0'}

    # Namespace
    ns='target'

    # PX4 SITL + Spawn x3
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
            'gz_model_name': model_name['gz_model_name'],
            'px4_autostart_id': autostart_id['px4_autostart_id'],
            'instance_id': instance_id['instance_id'],
            'xpos': xpos['xpos'],
            'ypos': ypos['ypos'],
            'zpos': zpos['zpos']
        }.items()
    )

    # MAVROS
    file_name = 'target_px4_pluginlists.yaml'
    package_share_directory = get_package_share_directory('smart_track')
    plugins_file_path = os.path.join(package_share_directory, file_name)
    file_name = 'target_px4_config.yaml'
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
            'tgt_system': '3',
            'fcu_url': 'udp://:14542@127.0.0.1:14559',
            'pluginlists_yaml': plugins_file_path,
            'config_yaml': config_file_path,
            'base_link_frame': 'target/base_link',
            'odom_frame': 'target/odom',
            'map_frame': 'map'
        }.items()
    )    

    # Static TF map(or world) -> local_pose_ENU
    map_frame = 'map'
    odom_frame= 'odom'
    map2pose_tf_node = Node(
        package='tf2_ros',
        name='map2px4_'+ns+'_tf_node',
        executable='static_transform_publisher',
        arguments=[str(xpos['xpos']), str(ypos['ypos']), str(zpos['zpos']), '0', '0', '0', map_frame, ns+'/'+odom_frame],
    )

    
    offboard_control_node = Node(
        package='smart_track',
        executable='offboard_control',
        output='screen',
        name='offboard_node',
        namespace=ns,
        parameters=[ {'trajectory_type': 'infty'},
                    {'system_id': 3},
                    {'radius': 3.0},
                    {'omega': 0.5},
                    {'normal_vector': [0.0, 0.0, 1.0]},
                    {'center': [10.0, 0.0, 10.0]},
        ],
        remappings=[
            ('mavros/state', 'mavros/state'),
            ('mavros/local_position/odom', 'mavros/local_position/odom'),
            ('mavros/setpoint_raw/local', 'mavros/setpoint_raw/local')
        ]
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
            'body_color': '[1.0, 0.0, 0.0, 1.0]',   # Set body_color directly
            'propeller_color': '[1.0, 1.0, 1.0, 1.0]',  # Set propeller_color directly
            'odom_topic': '/target/mavros/local_position/odom',     # Set odom_topic directly
        }.items(),
    )

    ld.add_action(gz_launch)
    # ld.add_action(px4_ros_node)
    ld.add_action(map2pose_tf_node)
    ld.add_action(offboard_control_node)
    ld.add_action(mavros_launch)
    # ld.add_action(quadcopter_marker_launch)

    return ld