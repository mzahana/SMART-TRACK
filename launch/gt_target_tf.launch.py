from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='smart_track',
            executable='gt_target_tf',
            name='gt_target_tf_node',
            output='screen',
            remappings=[
                ('/pose_array', '/yolo_detections_poses')
            ],
            parameters=[
                {'parent_frame': 'observer/odom'},
                {'child_frames': ['target/base_link']},
                {'publish_probability': 0.8},
                {'position_noise_std': 0.02},
                {'orientation_noise_std': 0.01},
                {'use_sim_time': True}
            ]
        )
    ])