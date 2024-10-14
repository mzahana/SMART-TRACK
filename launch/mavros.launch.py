#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    ld = LaunchDescription()

    fcu_url = LaunchConfiguration('fcu_url')
    gcs_url = LaunchConfiguration('gcs_url')
    tgt_system = LaunchConfiguration('tgt_system')
    tgt_component = LaunchConfiguration('tgt_component')
    log_output = LaunchConfiguration('log_output')
    pluginlists_yaml = LaunchConfiguration('pluginlists_yaml')
    config_yaml = LaunchConfiguration('config_yaml')
    fcu_protocol = LaunchConfiguration('fcu_protocol')
    mavros_namespace = LaunchConfiguration('mavros_namespace')
    base_link_frame = LaunchConfiguration("base_link_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    map_frame = LaunchConfiguration("map_frame")

    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='udp://:14540@127.0.0.1:14557',
        description="FCU interface Example /dev/ttyACM0:57600 OR udp://:14540@127.0.0.1:14557"
    )
    gcs_url_arg = DeclareLaunchArgument(
        'gcs_url',
        default_value='',
        description="GCS bridge interface"
    )
    tgt_system_arg = DeclareLaunchArgument(
        'tgt_system',
        default_value='1',
        description="MAVLink ID of the target system"
    )
    tgt_component_arg = DeclareLaunchArgument(
        'tgt_component',
        default_value='1',
        description="MAVLink component ID inside the target system"
    )
    log_output_arg = DeclareLaunchArgument(
        'log_output',
        default_value='screen',
        description="Type of log output"
    )
    package_name = 'mavros'
    file_name = 'launch/px4_pluginlists.yaml'
    package_share_directory = get_package_share_directory(package_name)
    file_path = os.path.join(package_share_directory, file_name)
    pluginlists_yaml_arg = DeclareLaunchArgument(
        'pluginlists_yaml',
        default_value=file_path,
        description="File path of the pluginlist_yaml file"
    )
    package_name = 'mavros'
    file_name = 'launch/px4_config.yaml'
    package_share_directory = get_package_share_directory(package_name)
    file_path = os.path.join(package_share_directory, file_name)
    config_yaml_arg = DeclareLaunchArgument(
        'config_yaml',
        default_value=file_path,
        description="File path of the mavros config_yaml file"
    )
    fcu_protocol_arg = DeclareLaunchArgument(
        'fcu_protocol',
        default_value='v2.0',
        description="MAVLink version"
    )
    namespace_arg = DeclareLaunchArgument(
        'mavros_namespace',
        default_value='mavros',
        description="Node namespace"
    )

    base_link_id_arg = DeclareLaunchArgument(
        'base_link_frame',
        default_value='base_link',
        description="base_link frame id"
    )

    odom_id_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='odom',
        description="odom frame id"
    )

    map_id_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description="map frame id"
    )

    

    # MAVROS node
    mavros_node = Node(
        namespace=mavros_namespace,
        package='mavros',
        executable='mavros_node',
        parameters=[
                {'fcu_url': fcu_url},
                {'gcs_url': gcs_url},
                {'tgt_system': tgt_system},
                {'tgt_component': tgt_component},
                {'fcu_protocol': fcu_protocol},
                {'base_link_frame_id': base_link_frame},
                {'odom_frame_id': odom_frame},
                {'map_frame_id': map_frame},
                {"use_sim_time": True},
                LaunchConfiguration('pluginlists_yaml'),
                LaunchConfiguration('config_yaml')
        ]
    )

    ld.add_action(fcu_url_arg)
    ld.add_action(gcs_url_arg)
    ld.add_action(tgt_component_arg)
    ld.add_action(tgt_system_arg)
    ld.add_action(log_output_arg)
    ld.add_action(fcu_protocol_arg)
    ld.add_action(namespace_arg)
    ld.add_action(pluginlists_yaml_arg)
    ld.add_action(config_yaml_arg)
    ld.add_action(mavros_node)
    ld.add_action(base_link_id_arg)
    ld.add_action(odom_id_arg)
    ld.add_action(map_id_arg)

    return ld