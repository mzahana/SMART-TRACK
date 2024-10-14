#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from std_msgs.msg import ColorRGBA
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import math

def quaternion_to_rotation_matrix(quat):
    x, y, z, w = quat
    # Compute rotation matrix from quaternion
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return rotation_matrix

def rotation_between_vectors(v1, v2):
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    # Compute cross product and dot product
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)
    # Compute skew-symmetric cross-product matrix
    skew_cross = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    # Compute rotation matrix
    if dot_prod < -0.9999999:
        # Vectors are opposite, rotate 180 degrees around any orthogonal vector
        orthogonal = np.array([1, 0, 0]) if abs(v1[0]) < 0.1 else np.array([0, 1, 0])
        rotation_matrix = quaternion_to_rotation_matrix(axis_angle_to_quaternion(orthogonal, math.pi))
    else:
        rotation_matrix = np.identity(3) + skew_cross + np.matmul(skew_cross, skew_cross) * ((1 - dot_prod)/(np.linalg.norm(cross_prod)**2))
    return rotation_matrix

def axis_angle_to_quaternion(axis, angle):
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    quat = np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(angle / 2)
    ])
    return quat

class QuadcopterMarkerPublisher(Node):
    def __init__(self):
        super().__init__('quadcopter_marker_publisher')
        # Declare parameters with default values
        self.declare_parameter('propeller_size', 0.1)  # Radius of the disk (meters)
        self.declare_parameter('arm_length', 0.5)      # From center to propeller center (meters)
        self.declare_parameter('body_size', [0.1, 0.1, 0.05])  # Body size (meters)
        self.declare_parameter('arm_thickness', 0.05)   # Arm thickness (meters)
        self.declare_parameter('body_color', [1.0, 0.0, 0.0, 1.0])  # RGBA for body
        self.declare_parameter('propeller_color', [0.0, 0.0, 1.0, 1.0])  # RGBA for propellers

        # Get parameters
        self.propeller_size = self.get_parameter('propeller_size').value
        self.arm_length = self.get_parameter('arm_length').value
        self.body_size = self.get_parameter('body_size').value
        self.arm_thickness = self.get_parameter('arm_thickness').value
        self.body_color = self.get_parameter('body_color').value
        self.propeller_color = self.get_parameter('propeller_color').value

        # Publisher for MarkerArray
        self.marker_pub = self.create_publisher(MarkerArray, 'quadcopter_marker', 10)

        # Subscriber to Odometry with sensor data QoS profile
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',  # Subscribe to 'odom', can be remapped
            self.odom_callback,
            qos_profile_sensor_data  # Use sensor data QoS profile
        )

        self.current_odom = None  # Store the latest odometry message

        # Timer to publish markers at a fixed rate
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def odom_callback(self, msg):
        self.current_odom = msg

    def timer_callback(self):
        if self.current_odom is None:
            return

        marker_array = MarkerArray()
        header = self.current_odom.header
        header.stamp = self.get_clock().now().to_msg()

        # Create the body marker
        body_marker = Marker()
        body_marker.header = header
        body_marker.ns = 'quadcopter_body'
        body_marker.id = 0
        body_marker.type = Marker.SPHERE
        body_marker.action = Marker.ADD
        body_marker.pose = self.current_odom.pose.pose
        body_marker.scale.x = self.body_size[0]  # Body size in x (meters)
        body_marker.scale.y = self.body_size[1]  # Body size in y (meters)
        body_marker.scale.z = self.body_size[2]  # Body size in z (meters)
        body_marker.color = ColorRGBA(
            r=self.body_color[0],
            g=self.body_color[1],
            b=self.body_color[2],
            a=self.body_color[3],
        )
        marker_array.markers.append(body_marker)

        # The propeller positions relative to the body center
        arm_offsets = [
            np.array([self.arm_length, 0, 0]),    # x+
            np.array([-self.arm_length, 0, 0]),   # x-
            np.array([0, self.arm_length, 0]),    # y+
            np.array([0, -self.arm_length, 0]),   # y-
        ]

        # Get the orientation quaternion from odom
        orientation = self.current_odom.pose.pose.orientation
        quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        # Rotation matrix from the quaternion
        rot_matrix = quaternion_to_rotation_matrix(quat)

        # For each arm, create a cylinder for the arm and a disk for the propeller
        for i, offset in enumerate(arm_offsets):
            # Rotate the offset
            offset_rotated = rot_matrix.dot(offset)

            # Position of the propeller in the odom frame
            propeller_pos = np.array([
                self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y,
                self.current_odom.pose.pose.position.z
            ]) + offset_rotated

            # Create the arm marker
            arm_marker = Marker()
            arm_marker.header = header
            arm_marker.ns = 'quadcopter_arm'
            arm_marker.id = i
            arm_marker.type = Marker.CYLINDER
            arm_marker.action = Marker.ADD

            # Position: midpoint between center and propeller_pos
            center_pos = np.array([
                self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y,
                self.current_odom.pose.pose.position.z
            ])
            arm_midpoint = (center_pos + propeller_pos) / 2.0

            arm_marker.pose.position.x = arm_midpoint[0]
            arm_marker.pose.position.y = arm_midpoint[1]
            arm_marker.pose.position.z = arm_midpoint[2]

            # Orientation: along the vector from center to propeller_pos
            direction = propeller_pos - center_pos
            length = np.linalg.norm(direction)
            if length == 0:
                continue  # Avoid division by zero
            direction_norm = direction / length

            # Compute rotation to align z-axis with the direction vector
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction_norm)
            rotation_angle = np.arccos(np.dot(z_axis, direction_norm))

            if np.linalg.norm(rotation_axis) < 1e-6:
                # Direction is along z-axis
                arm_marker.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )
            else:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                s = math.sin(rotation_angle / 2)
                q = Quaternion(
                    x=rotation_axis[0] * s,
                    y=rotation_axis[1] * s,
                    z=rotation_axis[2] * s,
                    w=math.cos(rotation_angle / 2)
                )
                arm_marker.pose.orientation = q

            # Scale: length is distance between center and propeller_pos
            arm_marker.scale.x = self.arm_thickness  # Arm thickness (meters)
            arm_marker.scale.y = self.arm_thickness
            arm_marker.scale.z = length

            # Color same as body
            arm_marker.color = ColorRGBA(
                r=self.body_color[0],
                g=self.body_color[1],
                b=self.body_color[2],
                a=self.body_color[3],
            )
            marker_array.markers.append(arm_marker)

            # Create the propeller marker (disk)
            propeller_marker = Marker()
            propeller_marker.header = header
            propeller_marker.ns = 'quadcopter_propeller'
            propeller_marker.id = i
            propeller_marker.type = Marker.CYLINDER
            propeller_marker.action = Marker.ADD
            propeller_marker.pose.position.x = propeller_pos[0]
            propeller_marker.pose.position.y = propeller_pos[1]
            propeller_marker.pose.position.z = propeller_pos[2]

            # Orientation: aligned with the quadcopter's orientation
            propeller_marker.pose.orientation = self.current_odom.pose.pose.orientation

            # Scale: radius is propeller_size, height is small
            propeller_marker.scale.x = self.propeller_size * 2
            propeller_marker.scale.y = self.propeller_size * 2
            propeller_marker.scale.z = 0.01  # Thin disk (meters)

            # Color
            propeller_marker.color = ColorRGBA(
                r=self.propeller_color[0],
                g=self.propeller_color[1],
                b=self.propeller_color[2],
                a=self.propeller_color[3],
            )
            marker_array.markers.append(propeller_marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = QuadcopterMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()