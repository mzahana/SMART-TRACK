#!/usr/bin/env python3
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from .detection import DroneDetector

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import Pose as TF2Pose

from geometry_msgs.msg import PoseArray, PointStamped, Pose, TransformStamped
from tf2_geometry_msgs import do_transform_point, do_transform_pose

class DepthCameraNode(Node):

    def __init__(self):
        # @ Initiate the node
        super().__init__("depth_camera_node")
        # @ Initiate the CV bridge
        self.cv_bridge_ = CvBridge()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('area_bounds',[390,1000]),
                ('circular_bounds',[0.3, 0.99]),
                ('convexity_bounds', [0.7,0.1]),
                ('d_group_max', 50),
                ('min_group_size',4),
                ('max_cam_depth', 20.0),
                ('depth_scale_factor',1.0),
                ('depth_step', 2.0),
                ('debug', True),
                ('show_debug_images', True),
                ('publish_processed_images', True),
                ('reference_frame', 'map'),
            ]
        )

        # Declare ros params
        self.area_bounds_ = self.get_parameter('area_bounds').get_parameter_value().integer_array_value
        self.circ_bounds_ = self.get_parameter('circular_bounds').get_parameter_value().double_array_value
        self.conv_bounds_ = self.get_parameter('convexity_bounds').get_parameter_value().double_array_value
        self.d_group_max_ = self.get_parameter('d_group_max').get_parameter_value().integer_value
        self.min_group_size_ = self.get_parameter('min_group_size').get_parameter_value().integer_value
        self.max_cam_depth_ = self.get_parameter('max_cam_depth').get_parameter_value().double_value
        self.depth_scale_factor_ = self.get_parameter('depth_scale_factor').get_parameter_value().double_value
        self.depth_step_ = self.get_parameter('depth_step').get_parameter_value().double_value
        self.debug_ = self.get_parameter('debug').get_parameter_value().bool_value
        self.show_debug_images_ = self.get_parameter('show_debug_images').get_parameter_value().bool_value
        self.pub_processed_images_ =self.get_parameter('publish_processed_images').get_parameter_value().bool_value
        self.reference_frame_ =self.get_parameter('reference_frame').get_parameter_value().string_value

        # Initiate class member 'detector'
        # TODO group all parameters into a dictionary before passing it to DroneDetector()
        self.detector_= DroneDetector(self.area_bounds_,
                                      self.circ_bounds_,
                                      self.conv_bounds_,
                                      self.d_group_max_,
                                      self.min_group_size_,
                                      self.max_cam_depth_,
                                      self.depth_scale_factor_,
                                      self.depth_step_ ,
                                      self.debug_
                                      )

        # Subscribe to image topic
        self.image_sub_ = self.create_subscription(Image,"observer/depth_image",self.imageCallback,10)
        # Subscribe to camera info topic
        self.caminfo_sub_ = self.create_subscription(CameraInfo, 'observer/camera_info', self.caminfoCallback, 10)

        # Publish detections positions
        self.detections_pub_ = self.create_publisher(PoseArray,'detections_poses',10)
        # Publish image with overlayed detections
        self.img_pub_ = self.create_publisher( Image,'detections_image',10)

        # Ref: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_,self)

    def imageCallback(self, msg: Image):
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge_.imgmsg_to_cv2(msg, desired_encoding="32FC1")#"16UC1")
        except Exception as e:
            self.get_logger().error("ros_to_cv conversion error {}".format(e))
            return
        
        # self.get_logger().info("Max depth = {}. Min depth = {}".format( cv_image.max(), cv_image.min()))
        
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.reference_frame_,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {self.reference_frame_} to {msg.header.frame_id}: {ex}')
            return

        try:            
            # Pre-process depth image and extracts contours and their features
            valid_detections, valid_depths, detections_img = self.detector_.preProcessing(cv_image)
        except Exception as e:
            self.get_logger().error("Error in preProcessing: {}".format(e))
            return

        try:
            # 3D projections
            positions = self.detector_.depthTo3D(valid_detections, valid_depths)
            if self.debug_:
                self.get_logger().info("3D positions: {}".format(positions), throttle_duration_sec=1)
        except Exception as e:
            self.get_logger().error("Error in depthTo3D: {}".format(e))
            return

        try:
            pose_array = PoseArray()
            pose_array = self.transformPositions(positions, self.reference_frame_, 
                                                 msg.header.frame_id, 
                                                 msg.header.stamp, transform)
        except Exception as e:
            self.get_logger().error("Error in transforming positions: {}".format(e))
            return

        if len(pose_array.poses) > 0:
            self.detections_pub_.publish(pose_array)

        if(self.pub_processed_images_):
            ros_img = self.cv_bridge_.cv2_to_imgmsg(detections_img)
            self.img_pub_.publish(ros_img)
                

    def caminfoCallback(self,msg: CameraInfo):
        # TODO : fill self.camera_info_ field
        P = np.array(msg.p)
        K = np.array(msg.k)
        # if len(P) == 12: # Sanity check
        #     P = P.reshape((3,4))
        #     self.detector_.camera_info_ = {'fx': P[0][0], 'fy': P[1][1], 'cx': P[0][2], 'cy': P[1][2]}

        if len(K) == 9: # Sanity check
            K = K.reshape((3,3))
            self.detector_.camera_info_ = {'fx': K[0][0], 'fy': K[1][1], 'cx': K[0][2], 'cy': K[1][2]}

    def transformPositions(self, positions: list, parent_frame: str, child_frame: str, tf_time, tr: TransformStamped) -> PoseArray:
        """
        @brief Converts 3D positions in the camera frame to the parent_frame
        @param positions: List of 3D positions in the child frame (sensor e.g. camera)
        @param parent_frame: Frame to transform positions to
        @param child_frame: Current frame of positions
        @param tf_time: Time at which positions were computed
        @param tr: Transform 4x4 matrix theat encodes rotation and translation
        @return pose_array: PoseArray of all transformed positions
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = parent_frame
        pose_array.header.stamp = tf_time
        
        for pos in positions:            
            tf2_pose_msg = TF2Pose()
            tf2_pose_msg.position.x = pos[0]
            tf2_pose_msg.position.y = pos[1]
            tf2_pose_msg.position.z = pos[2]
            tf2_pose_msg.orientation.w = 1.0        
            
            try:
                transformed_pose = do_transform_pose(tf2_pose_msg, tr)
                pose_array.poses.append(transformed_pose)
            except Exception as e:
                self.get_logger().error("Error in transforming point {}".format(e))
                continue

            

        return pose_array


def main(args=None):
    rclpy.init(args=args)
    depth_camera_node = DepthCameraNode()
    depth_camera_node.get_logger().info("Drone detection node has started")
    rclpy.spin(depth_camera_node)
    depth_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()