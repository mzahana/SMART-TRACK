from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Add argument for the path of the yaml file
    detection_yaml = LaunchConfiguration('detection_yaml')
    depth_topic = LaunchConfiguration('depth_topic')
    caminfo_topic = LaunchConfiguration('caminfo_topic')
    detections_topic = LaunchConfiguration('detections_topic')
    namespace = LaunchConfiguration('detector_ns')

    config = os.path.join(
        get_package_share_directory('smart_track'),
        'detection_param.yaml'
    )
    
    detection_yaml_launch_arg = DeclareLaunchArgument(
        'detection_yaml',
        default_value=config
    )

    depth_topic_launch_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='observer/depth_image'
    )

    caminfo_topic_launch_arg = DeclareLaunchArgument(
        'caminfo_topic',
        default_value='observer/camera_info'
    )

    detections_topic_launch_arg = DeclareLaunchArgument(
        'detections_topic',
        default_value='detections_poses'
    )

    namespace_launch_arg = DeclareLaunchArgument(
        'detector_ns',
        default_value=''
    )

    # Detection node
    detection_node = Node(
        package='smart_track',
        executable='detection_node',
        name='detection_node',
        namespace=namespace,
        output='screen',
        parameters=[detection_yaml],
        remappings=[('observer/depth_image', depth_topic),
                    ('observer/camera_info', caminfo_topic),
                    ('detections_poses', detections_topic)
                    ]
    )

    ld = LaunchDescription()

    ld.add_action(detection_yaml_launch_arg)
    ld.add_action(depth_topic_launch_arg)
    ld.add_action(caminfo_topic_launch_arg)
    ld.add_action(namespace_launch_arg)
    ld.add_action(detections_topic_launch_arg)
    ld.add_action(detection_node)

    return ld