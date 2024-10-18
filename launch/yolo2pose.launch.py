from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    depth_topic = LaunchConfiguration('depth_topic')
    caminfo_topic = LaunchConfiguration('caminfo_topic')
    detections_poses_topic = LaunchConfiguration('detections_poses_topic')
    yolo_detections_topic = LaunchConfiguration('yolo_detections_topic')
    namespace = LaunchConfiguration('detector_ns')
    debug = LaunchConfiguration('debug')
    reference_frame = LaunchConfiguration('reference_frame')

    debug_launch_arg = DeclareLaunchArgument(
        'debug',
        default_value='True'
    )

    depth_topic_launch_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='observer/depth_image'
    )

    caminfo_topic_launch_arg = DeclareLaunchArgument(
        'caminfo_topic',
        default_value='observer/camera_info'
    )

    detections_poses_topic_launch_arg = DeclareLaunchArgument(
        'detections_poses_topic',
        default_value='yolo_detections_poses'
    )

    yolo_detections_topic_launch_arg = DeclareLaunchArgument(
        'yolo_detections_topic',
        default_value='detections'
    )

    namespace_launch_arg = DeclareLaunchArgument(
        'detector_ns',
        default_value=''
    )

    reference_frame_launch_arg = DeclareLaunchArgument(
        'reference_frame',
        default_value='world'
    )

    # yolo2pose_node node
    detection_node = Node(
        package='smart_track',
        executable='yolo2pose_node',
        name='yolo2pose_node',
        namespace=namespace,
        output='screen',
        parameters=[{'debug': debug},
                    {'reference_frame': reference_frame},
                    {'use_sim_time': True},],

        remappings=[('observer/depth_image', depth_topic),
                    ('observer/camera_info', caminfo_topic),
                    ('yolo_poses', detections_poses_topic),
                    ('detections', yolo_detections_topic)
                    ]
    )

    ld = LaunchDescription()

    ld.add_action(depth_topic_launch_arg)
    ld.add_action(yolo_detections_topic_launch_arg)
    ld.add_action(debug_launch_arg)
    ld.add_action(reference_frame_launch_arg)
    ld.add_action(caminfo_topic_launch_arg)
    ld.add_action(namespace_launch_arg)
    ld.add_action(detections_poses_topic_launch_arg)
    ld.add_action(detection_node)

    return ld