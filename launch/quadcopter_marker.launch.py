#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments (parameters)
    propeller_size_arg = DeclareLaunchArgument(
        'propeller_size',
        default_value='0.1',
        description='Radius of the propeller disks (meters)'
    )

    arm_length_arg = DeclareLaunchArgument(
        'arm_length',
        default_value='0.5',
        description='Length from center to propeller center (meters)'
    )

    body_color_arg = DeclareLaunchArgument(
        'body_color',
        default_value='[1.0, 0.0, 0.0, 1.0]',
        description='RGBA color of the quadcopter body'
    )

    propeller_color_arg = DeclareLaunchArgument(
        'propeller_color',
        default_value='[0.0, 0.0, 1.0, 1.0]',
        description='RGBA color of the propellers'
    )

    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='odom',
        description='Odometry topic name to remap'
    )

    node_ns_arg = DeclareLaunchArgument(
        'node_ns',
        default_value='',
        description='Node namespace'
    )

    # Define the node with parameters and remappings
    quadcopter_marker_node = Node(
        package='smart_track',  # Replace with your actual package name
        executable='drone_marker_node',
        name='quadcopter_marker_publisher',
        namespace=LaunchConfiguration('node_ns'),
        output='screen',
        parameters=[{
            'propeller_size': LaunchConfiguration('propeller_size'),
            'arm_length': LaunchConfiguration('arm_length'),
            'body_color': LaunchConfiguration('body_color'),
            'propeller_color': LaunchConfiguration('propeller_color'),
        }],
        remappings=[
            ('odom', LaunchConfiguration('odom_topic')),  # Remap 'odom' topic
        ]
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the launch arguments
    ld.add_action(propeller_size_arg)
    ld.add_action(arm_length_arg)
    ld.add_action(body_color_arg)
    ld.add_action(propeller_color_arg)
    ld.add_action(odom_topic_arg)
    ld.add_action(node_ns_arg)

    # Add the node to the launch description
    ld.add_action(quadcopter_marker_node)

    return ld