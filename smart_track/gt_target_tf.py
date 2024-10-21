#!/usr/bin/env python3

"""
@Description
This node publishes ground truth measurement of the target drone with optional noise and probabilistic publishing.
This can be used to test the Kalman filter performance, as well as all the subsequent modules (prediction and tracking).
Subscribes to TF tree, and finds the transformation from map frame to the target/base_link.
Then publishes pose measurements as geometry_msgs/msg/PoseArray.

Author: Mohamed Abdelkader
Copyright 2023
Contact: mohamedashraf123@gmail.com

"""
import rclpy
from rclpy.node import Node

from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PoseArray, Pose
from geometry_msgs.msg import TransformStamped

import random
import numpy as np

class TFLookupNode(Node):

    def __init__(self):
        super().__init__('tf_lookup_node')
        self.declare_parameter('parent_frame', 'base_link')
        self.declare_parameter('child_frames', ['base_link'])
        self.declare_parameter('publish_probability', 0.8)
        self.declare_parameter('position_noise_std', 0.0)
        self.declare_parameter('orientation_noise_std', 0.0)

        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child_frames = self.get_parameter('child_frames').get_parameter_value().string_array_value
        self.publish_probability = self.get_parameter('publish_probability').get_parameter_value().double_value
        self.position_noise_std = self.get_parameter('position_noise_std').get_parameter_value().double_value
        self.orientation_noise_std = self.get_parameter('orientation_noise_std').get_parameter_value().double_value

        self.publisher_ = self.create_publisher(PoseArray, 'pose_array', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        if random.random() > self.publish_probability:
            return  # Skip publishing based on probability

        pose_array = PoseArray()
        pose_array.header.frame_id = self.parent_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for child_frame in self.child_frames:
            try:
                transform = self.tf_buffer.lookup_transform(self.parent_frame, child_frame, rclpy.time.Time())
                pose = self.transform_to_pose(transform)
                pose_array.poses.append(pose)
            except Exception as e:
                self.get_logger().warn(f"Failed to get transform from {self.parent_frame} to {child_frame}: {str(e)}")

        self.publisher_.publish(pose_array)

    def transform_to_pose(self, transform: TransformStamped) -> Pose:
        pose = Pose()
        pose.position.x = transform.transform.translation.x + random.gauss(0, self.position_noise_std)
        pose.position.y = transform.transform.translation.y + random.gauss(0, self.position_noise_std)
        pose.position.z = transform.transform.translation.z + random.gauss(0, self.position_noise_std)

        pose.orientation.x = transform.transform.rotation.x + random.gauss(0, self.orientation_noise_std)
        pose.orientation.y = transform.transform.rotation.y + random.gauss(0, self.orientation_noise_std)
        pose.orientation.z = transform.transform.rotation.z + random.gauss(0, self.orientation_noise_std)
        pose.orientation.w = transform.transform.rotation.w + random.gauss(0, self.orientation_noise_std)
        return pose


def main(args=None):
    rclpy.init(args=args)

    node = TFLookupNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
