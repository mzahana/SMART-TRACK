import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path

from .trajectories import Circle3D, Infinity3D

from visualization_msgs.msg import Marker

from mavros_msgs.msg import State, PositionTarget


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_contorl_node')

        self.declare_parameter('system_id', 1)
        self.sys_id_ = self.get_parameter('system_id').get_parameter_value().integer_value

        self.declare_parameter('radius', 5.0)
        self.radius_ = self.get_parameter('radius').get_parameter_value().double_value

        self.declare_parameter('omega', 0.5)
        self.omega_ = self.get_parameter('omega').get_parameter_value().double_value

        self.declare_parameter('trajectory_type', 'circle')
        self.trajectory_type_ = self.get_parameter('trajectory_type').get_parameter_value().string_value

        self.declare_parameter('normal_vector', [0., 0., 1.])
        self.normal_vector_ = self.get_parameter('normal_vector').get_parameter_value().double_array_value

        self.declare_parameter('center', [0., 0., 1.])
        self.center_ = self.get_parameter('center').get_parameter_value().double_array_value

        # Initialize the trajectory generator based on the selected type
        if self.trajectory_type_== 'circle':
            self.trajectory_generator_ = Circle3D(np.array(self.normal_vector_), np.array(self.center_), radius=self.radius_, omega=self.omega_)
        elif self.trajectory_type_ == 'infty':
            self.trajectory_generator_ = Infinity3D(np.array(self.normal_vector_), np.array(self.center_), radius=self.radius_, omega=self.omega_)
        else:
            raise ValueError("Invalid trajectory type. Supported types are 'circle', 'infty'.")
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_volatile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_ = Odometry() # latest odom

        self.status_sub_ = self.create_subscription(
            State,
            'mavros/state',
            self.vehicleStatusCallback,
            qos_profile_transient)
        
        self.odom_sub_ = self.create_subscription(
            Odometry,
            'mavros/local_position/odom',
            self.odomCallback,
            qos_profile_sensor_data)
        
        self.vehicle_path_pub_ = self.create_publisher(Path, 'offboard_visualizer/vehicle_path', 10)
        self.setpoint_path_pub_ = self.create_publisher(Path, 'offboard_visualizer/setpoint_path', 10)

        self.setopint_pub_ = self.create_publisher(PositionTarget, 'mavros/setpoint_raw/local', qos_profile_sensor_data)

        timer_period = 0.02  # seconds
        self.cmd_timer_ = self.create_timer(timer_period, self.cmdloopCallback)

        self.offboard_setpoint_counter_ = 0

        self.is_armed_ = False
        self.dt_ = timer_period

        self.vehicle_path_msg_ = Path()
        self.setpoint_path_msg_ = Path()


    def vehicleStatusCallback(self, msg: State):
        self.is_armed_ = msg.armed

    def odomCallback(self, msg: Odometry):
        self.odom_ = msg

    def create_arrow_marker(self, id, tail, vector):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = self.odom_.header.frame_id
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = 'arrow'
        msg.id = id
        msg.type = Marker.ARROW
        msg.scale.x = 0.1
        msg.scale.y = 0.2
        msg.scale.z = 0.0
        msg.color.r = 0.5
        msg.color.g = 0.5
        msg.color.b = 0.0
        msg.color.a = 1.0
        dt = 0.3
        tail_point = Point()
        tail_point.x = tail[0]
        tail_point.y = tail[1]
        tail_point.z = tail[2]
        head_point = Point()
        head_point.x = tail[0] + dt * vector[0]
        head_point.y = tail[1] + dt * vector[1]
        head_point.z = tail[2] + dt * vector[2]
        msg.points = [tail_point, head_point]
        return msg

   
    def cmdloopCallback(self):

        t_now = Clock().now()
        point = self.trajectory_generator_.generate_trajectory_setpoint(t_now.nanoseconds / 1000/1000/1000)

        setpoint_msg = PositionTarget()
        setpoint_msg.header.stamp = self.get_clock().now().to_msg()
        setpoint_msg.header.frame_id = self.odom_.header.frame_id
        setpoint_msg.coordinate_frame= PositionTarget.FRAME_LOCAL_NED
        setpoint_msg.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                    PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ+ PositionTarget.IGNORE_YAW_RATE
        setpoint_msg.position.x = point[0]
        setpoint_msg.position.y = point[1]
        setpoint_msg.position.z = point[2]
        # yaw  = atan(d_y, d_x)
        yaw = np.arctan2(point[1] - self.odom_.pose.pose.position.y, point[0] - self.odom_.pose.pose.position.x)
        setpoint_msg.yaw = yaw

        self.setopint_pub_.publish(setpoint_msg)
        
         # Publish time history of the vehicle path
        vehicle_pose_msg = PoseStamped()
        vehicle_pose_msg.header = self.odom_.header
        vehicle_pose_msg.pose.position = self.odom_.pose.pose.position
        vehicle_pose_msg.pose.orientation = self.odom_.pose.pose.orientation
        self.vehicle_path_msg_.header = self.odom_.header
        self.vehicle_path_msg_.poses.append(vehicle_pose_msg)
        if (len(self.vehicle_path_msg_.poses) > 500):
            self.vehicle_path_msg_.poses.pop(0)

        self.vehicle_path_pub_.publish(self.vehicle_path_msg_)

        # Publish time history of the vehicle path
        setpoint_pose_msg = PoseStamped()
        setpoint_pose_msg.header.frame_id = self.odom_.header.frame_id
        setpoint_pose_msg.header.stamp = self.get_clock().now().to_msg()
        setpoint_pose_msg.pose.position.x = point[0]
        setpoint_pose_msg.pose.position.y = point[1]
        setpoint_pose_msg.pose.position.z = point[2]
        self.setpoint_path_msg_.header = setpoint_pose_msg.header
        self.setpoint_path_msg_.poses.append(setpoint_pose_msg)
        if (len(self.setpoint_path_msg_.poses) > 500):
            self.setpoint_path_msg_.poses.pop(0)

        self.setpoint_path_pub_.publish(self.setpoint_path_msg_)

def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()