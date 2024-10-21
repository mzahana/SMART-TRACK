#!/usr/bin/env python3

"""
Yolo2PoseNode

This node receives:
    - vision_msgs/msg/Detection2dArray msg
    - Depth image msg sensor_msgs/msg/Image

And converts the 2D YOLO detections to 3D positions as:
    - geometry_msgs/msg/PoseArray
    
Author: Mohamed Abdelkader, Khaled Gabr
Contact: mohamedashraf123@gmail.com
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from yolov8_msgs.msg import DetectionArray
from multi_target_kf.msg import KFTracks
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import Pose as TF2Pose
from tf2_geometry_msgs import do_transform_pose, do_transform_pose_with_covariance_stamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
import copy

class Yolo2PoseNode(Node):

    def __init__(self):
        super().__init__("yolo2pose_node")

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('debug', True),
                ('publish_processed_images', True),
                ('reference_frame', 'map'),
                ('camera_frame', 'x500_d435_1/link/realsense_d435'),
                ('yolo_measurement_only', True),
                ('kf_feedback', True),
                ('depth_roi', 5.0),
                ('std_range', 5.0),
            ]
        )

        # Get parameters
        self.debug_ = self.get_parameter('debug').value
        self.publish_processed_images_ = self.get_parameter('publish_processed_images').value
        self.reference_frame_ = self.get_parameter('reference_frame').value
        self.camera_frame_ = self.get_parameter('camera_frame').value

        self.cv_bridge_ = CvBridge()

        # Camera intrinsics
        self.camera_info_ = None

        # TF buffer and listener
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # Initialize variables
        self.latest_detections_msg_ = DetectionArray()
        self.latest_kftracks_msg_ = KFTracks()
        self.latest_depth_synced_with_yolo_msg_ = Image()
        self.latest_depth_synced_with_kf_msg_ = Image()
        self.latest_detection_time_ = 0.0
        self.latest_kftracks_time_ = 0.0
        self.last_detection_t_ = 0.0
        self.last_kf_measurements_t_ = 0.0
        self.new_measurements_yolo = False
        self.new_measurements_kf = False

        # Subscribers using message_filters
        self.depth_sub_ = Subscriber(self, Image, "observer/depth_image")
        self.detections_sub_ = Subscriber(self, DetectionArray, "detections")
        self.kftracks_sub_ = Subscriber(self, KFTracks, "kf/good_tracks")

        # Synchronizers
        self.detection_depth_sync = ApproximateTimeSynchronizer(
            [self.detections_sub_, self.depth_sub_], queue_size=100, slop=0.1)
        self.detection_depth_sync.registerCallback(self.detection_depth_callback)

        self.kftracks_depth_sync = ApproximateTimeSynchronizer(
            [self.kftracks_sub_, self.depth_sub_], queue_size=100, slop=0.1)
        self.kftracks_depth_sync.registerCallback(self.kftracks_depth_callback)

        # Camera info subscriber
        self.caminfo_sub_ = self.create_subscription(
            CameraInfo, 'observer/camera_info', self.caminfoCallback, 10)

        # Timer for state machine
        self.timer_ = self.create_timer(0.05, self.timer_callback)  # Adjust interval as needed

        # Publishers
        self.poses_pub_ = self.create_publisher(PoseArray, 'yolo_poses', 10)
        self.overlay_ellipses_image_yolo_ = self.create_publisher(Image, "overlay_yolo_image", 10)

        # Initialize variables for processing
        self.latest_pixels_ = []
        self.latest_covariances_2d_ = []
        self.latest_depth_ranges_ = []

        self.filter_kernel_size = (5, 5)
        self.depth_threshold = 0

    def detection_depth_callback(self, detections_msg, depth_msg):
        """
        Callback for synchronized detections and depth images.
        """
        # Process synchronized depth and detection messages
        self.latest_detections_msg_ = detections_msg
        self.latest_depth_synced_with_yolo_msg_ = depth_msg
        self.latest_detection_time_ = self.get_clock().now()
        # self.update_detections(detections_msg)

    def kftracks_depth_callback(self, kftracks_msg, depth_msg):
        """
        Callback for synchronized KFTracks and depth images.
        """
        # Process synchronized depth and KF tracks messages
        self.latest_kftracks_msg_ = kftracks_msg
        self.latest_depth_synced_with_kf_msg_ = depth_msg
        self.latest_kftracks_time_ = self.get_clock().now()
        # self.update_kf_tracks(kftracks_msg)

    def is_new_detections(self):
        """
        Update function for detections.
        """
        current_detection_t = float(self.latest_detections_msg_.header.stamp.sec) + \
                              float(self.latest_detections_msg_.header.stamp.nanosec) / 1e9

        if len(self.latest_detections_msg_.detections) > 0:
            if current_detection_t > self.last_detection_t_:
                self.last_detection_t_ = current_detection_t
                return True
            else:
                return False
        else:
            return False

    def is_new_kf_tracks(self):
        """
        Update function for KF tracks.
        """
        current_kf_measurement_t = float(self.latest_kftracks_msg_.header.stamp.sec) + \
                                   float(self.latest_kftracks_msg_.header.stamp.nanosec) / 1e9

        if len(self.latest_kftracks_msg_.tracks) > 0:
            if current_kf_measurement_t > self.last_kf_measurements_t_ and  current_kf_measurement_t > self.last_detection_t_:
                self.last_kf_measurements_t_ = current_kf_measurement_t
                return True
            else:
                return False
        else:
            False

    def timer_callback(self):
        """
        Timer callback acting as a state machine to decide whether to use YOLO or KF measurements.
        """
        use_yolo = self.get_parameter('yolo_measurement_only').value
        use_kf = self.get_parameter('kf_feedback').value

        if use_yolo :
            if self.is_new_detections():
                yolo_poses = self.yolo_process_pose(self.latest_depth_synced_with_yolo_msg_, self.latest_detections_msg_)
                if yolo_poses and len(yolo_poses.poses) > 0:
                    self.poses_pub_.publish(yolo_poses)
                    return
                else:
                    self.get_logger().warn("[Yolo2PoseNode::timer_callback] Got a new Yolo measurment, but could not compute new poses!")
            # else:
            #     self.get_logger().warn("[Yolo2PoseNode::timer_callback] No new YOLO detections!")

        if use_kf:
            if self.is_new_kf_tracks():
                kf_poses = self.kf_process_pose(self.latest_depth_synced_with_kf_msg_, self.latest_kftracks_msg_)
                if kf_poses and len(kf_poses.poses) > 0:
                    self.poses_pub_.publish(kf_poses)
                    return
                else:
                    self.get_logger().warn("[Yolo2PoseNode::timer_callback] Got new KF tracks, but could not compute new poses!")
            # else:
            #     self.get_logger().warn("[Yolo2PoseNode::timer_callback] No new KF Tracks!")

        if not use_yolo and not use_kf:
            self.get_logger().warn("[Yolo2PoseNode::timer_callback] use_yolo and use_kf are False")
    
    def caminfoCallback(self, msg: CameraInfo):
        """
        Callback function for handling camera information.
        """
        # Fill self.camera_info_ field
        K = np.array(msg.k)
        if len(K) == 9:
            K = K.reshape((3, 3))
            self.camera_info_ = {'fx': K[0][0], 'fy': K[1][1], 'cx': K[0][2], 'cy': K[1][2]}
        else:
            self.get_logger().warn("[Yolo2PoseNode::caminfoCallback] Invalid camera info received.")

    def yolo_process_pose(self, depth_msg: Image, yolo_msg: DetectionArray):
        """
        Processes YOLO detections in the provided depth image to extract object poses.
        """
        if self.camera_info_ is None:
            if self.debug_:
                self.get_logger().warn("[Yolo2PoseNode::yolo_process_pose] camera_info is None. Return")
            return None

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error("[Yolo2PoseNode::yolo_process_pose] Image to CvImg conversion error {}".format(e))
            return None

        try:

            transform = self.tf_buffer_.lookup_transform(
                self.reference_frame_,
                depth_msg.header.frame_id,
                rclpy.time.Time(seconds=0),  # Use time=0 to get the latest transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # You can still specify a timeout
            )
        except TransformException as ex:
            self.get_logger().error(
                f'[Yolo2PoseNode::yolo_process_pose] Could not transform {self.reference_frame_} to {depth_msg.header.frame_id}: {ex}')
            return None

        poses_msg = PoseArray()
        poses_msg.header = copy.deepcopy(yolo_msg.header)
        poses_msg.header.frame_id = self.reference_frame_
        ellipse_color = (0, 255, 0)
        text_color = (0, 255, 0)

        for obj in yolo_msg.detections:
            x = int(obj.bbox.center.position.x - obj.bbox.size.x / 2)
            y = int(obj.bbox.center.position.y - obj.bbox.size.y / 2)
            w = int(obj.bbox.size.x)
            h = int(obj.bbox.size.y)
            self.filter_kernel_size = (5, 5)
            self.depth_threshold = 0
            depth_image_roi = cv_image[y:y + h, x:x + w]

            if depth_image_roi.size == 0:
                self.get_logger().warn("[Yolo2PoseNode::yolo_process_pose] The bounding box from Yolo has no pixels. Skipping")
                continue

            _, depth_thresholded = cv2.threshold(depth_image_roi, self.depth_threshold, 255, cv2.THRESH_BINARY)
            depth_filtered = cv2.GaussianBlur(depth_thresholded, self.filter_kernel_size, 0)
            contours, _ = cv2.findContours(depth_filtered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Interested in the largest contour only:
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:  # Avoid division by zero
                    cx = int(M["m10"] / M["m00"])  # Centroid x
                    cy = int(M["m01"] / M["m00"])  # Centroid y
                else:
                    if self.debug_:
                        self.get_logger().warn("[Yolo2PoseNode::yolo_process_pose] Moment computation resulted in division by zero")
                    continue
                depth_at_centroid = depth_image_roi[cy, cx]

                # Use centroid pixel and depth_at_centroid for further processing:
                pixel = [x + cx, y + cy]
                pose_msg = self.depthToPoseMsg(pixel, depth_at_centroid)
                transformed_pose_msg = self.transform_pose(pose_msg, transform)

                if transformed_pose_msg is not None:
                    poses_msg.poses.append(transformed_pose_msg)

                # Drawing circle for this detection
                center_coordinates = (int(obj.bbox.center.position.x), int(obj.bbox.center.position.y))
                cv2.circle(cv_image, center_coordinates, int(w / 2), ellipse_color, 1)

        cv2.putText(cv_image, "YOLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        image_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.overlay_ellipses_image_yolo_.publish(image_msg)
        return poses_msg

    def kf_process_pose(self, depth_msg: Image, kf_msg: KFTracks):
        """
        Processes Kalman Filter tracks in the provided depth image to extract object poses.
        """
        if self.camera_info_ is None:
            if self.debug_:
                self.get_logger().warn("[Yolo2PoseNode::kf_process_pose] camera_info is None. Return")
            return None

        depth_roi_ = self.get_parameter('depth_roi').value
        std_range_ = self.get_parameter('std_range').value

        try:
            transform = self.tf_buffer_.lookup_transform(
                self.reference_frame_,
                depth_msg.header.frame_id,
                rclpy.time.Time(seconds=0),  # Use time=0 to get the latest transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # You can still specify a timeout
            )
        except TransformException as ex:
            self.get_logger().error(f'[kf_process_pose] Could not transform {self.reference_frame_} to {depth_msg.header.frame_id}: {ex}')
            return None

        poses_msg_kf = PoseArray()
        poses_msg_kf.header = copy.deepcopy(depth_msg.header)
        poses_msg_kf.header.frame_id = self.reference_frame_

        depth_image_cv = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        image_width = depth_image_cv.shape[1]
        image_height = depth_image_cv.shape[0]

        self.latest_pixels_, self.latest_covariances_2d_, self.latest_depth_ranges_ = self.process_and_store_track_data(kf_msg)

        for mean_pixel, covariance_matrix, depth_range in zip(self.latest_pixels_, self.latest_covariances_2d_, self.latest_depth_ranges_):
            x, y = mean_pixel

            if 0 <= x < image_width and 0 <= y < image_height:
                # Calculate ellipse parameters based on the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix[:2, :2])
                if np.any(eigenvalues < 0):
                    self.get_logger().warn("Covariance matrix has negative eigenvalues.")
                    continue
                rotation_angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                axes_lengths = (int(depth_roi_ * np.sqrt(eigenvalues[0])), int(depth_roi_ * np.sqrt(eigenvalues[1])))

                # Draw the ellipse on the depth image
                cv2.ellipse(depth_image_cv, (x, y), axes_lengths, rotation_angle, 0, 360, (0, 255, 0), 2)

                # Perform depth-based filtering
                depth_image_blurred = cv2.GaussianBlur(depth_image_cv, (5, 5), 0)
                depth_mask = cv2.inRange(depth_image_blurred, depth_range[0], depth_range[1])

                kfcontours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                nearest_depth_value = None
                min_distance = float('inf')
                nearest_centroid_x = 0.0
                nearest_centroid_y = 0.0

                for kfcontour in kfcontours:
                    contour_moments = cv2.moments(kfcontour)
                    if contour_moments["m00"] != 0:
                        centroid_x = int(contour_moments["m10"] / contour_moments["m00"])
                        centroid_y = int(contour_moments["m01"] / contour_moments["m00"])
                        if 0 <= centroid_x < image_width and 0 <= centroid_y < image_height:
                            contour_depth_values = depth_image_cv[kfcontour[:, :, 1], kfcontour[:, :, 0]]
                            valid_depth_indices = np.logical_and(
                                depth_range[0] <= contour_depth_values,
                                contour_depth_values <= depth_range[1]
                            )

                            if np.any(valid_depth_indices):
                                average_depth = np.mean(contour_depth_values[valid_depth_indices])
                                distance = np.sqrt((mean_pixel[0] - centroid_x) ** 2 + (mean_pixel[1] - centroid_y) ** 2)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_depth_value = average_depth
                                    nearest_centroid_x = centroid_x
                                    nearest_centroid_y = centroid_y

                if nearest_depth_value is not None:
                    pixel_pose = [nearest_centroid_x, nearest_centroid_y]
                    kf_pose_msg = self.depthToPoseMsg(pixel_pose, nearest_depth_value)
                    kf_transformed_pose_msg = self.transform_pose(kf_pose_msg, transform)
                    if kf_transformed_pose_msg is not None:
                        poses_msg_kf.poses.append(kf_transformed_pose_msg)
                else:
                    self.get_logger().warn("No valid depth value found for KF tracks.")

        cv2.putText(depth_image_cv, "KF", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Publish the modified depth image with ellipses
        ellipses_image_msg = self.cv_bridge_.cv2_to_imgmsg(depth_image_cv, encoding="passthrough")
        self.overlay_ellipses_image_yolo_.publish(ellipses_image_msg)

        return poses_msg_kf

    def process_and_store_track_data(self, kf_msg: KFTracks):
        """
        Processes Kalman Filter track data to extract pixel coordinates, 2D covariances, and depth ranges.
        """
        self.latest_pixels_.clear()
        self.latest_covariances_2d_.clear()
        self.latest_depth_ranges_.clear()

        try:
            transform = self.tf_buffer_.lookup_transform(
                self.camera_frame_,
                kf_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().error(
                f'[process_and_store_track_data] Could not transform {kf_msg.header.frame_id} to {self.camera_frame_}: {ex}')
            return [], [], []

        std_range_ = self.get_parameter('std_range').value

        for track in kf_msg.tracks:
            x = track.pose.pose.position.x
            y = track.pose.pose.position.y
            z = track.pose.pose.position.z

            covariance = track.pose.covariance
            cov_x = covariance[0]
            cov_y = covariance[7]
            cov_z = covariance[14]

            tf2_cam_msg = PoseWithCovarianceStamped()
            tf2_cam_msg.header = kf_msg.header  # Ensure the header is set correctly
            tf2_cam_msg.pose.pose.position.x = x
            tf2_cam_msg.pose.pose.position.y = y
            tf2_cam_msg.pose.pose.position.z = z
            tf2_cam_msg.pose.pose.orientation.w = 1.0
            tf2_cam_msg.pose.covariance = [0.0] * 36
            tf2_cam_msg.pose.covariance[0] = cov_x
            tf2_cam_msg.pose.covariance[7] = cov_y
            tf2_cam_msg.pose.covariance[14] = cov_z

            transformed_pose_msg = self.transform_pose_cov(tf2_cam_msg, transform)

            if transformed_pose_msg:
                x_transformed = transformed_pose_msg.pose.pose.position.x
                y_transformed = transformed_pose_msg.pose.pose.position.y
                z_transformed = transformed_pose_msg.pose.pose.position.z
                pixel = self.project_3d_to_2d(x_transformed, y_transformed, z_transformed)

                cov_transformed = transformed_pose_msg.pose.covariance
                cov_x_transformed = cov_transformed[0]
                cov_y_transformed = cov_transformed[7]
                cov_z_transformed = cov_transformed[14]
                covariance_2d = self.project_3d_covariance_to_2d(
                    x_transformed, y_transformed, z_transformed,
                    cov_x_transformed, cov_y_transformed, cov_z_transformed
                )

                if cov_z_transformed < 0:
                    self.get_logger().warn("Negative variance in Z after transformation.")
                    continue

                depth_range = (
                    max(0, z_transformed - std_range_ * np.sqrt(cov_z_transformed)),
                    z_transformed + std_range_ * np.sqrt(cov_z_transformed)
                )

                self.latest_depth_ranges_.append(depth_range)
                self.latest_pixels_.append(pixel)
                self.latest_covariances_2d_.append(covariance_2d)

        return self.latest_pixels_, self.latest_covariances_2d_, self.latest_depth_ranges_

    def project_3d_to_2d(self, x_cam, y_cam, z_cam):
        """
        Projects 3D coordinates onto 2D pixel coordinates.
        """
        pixel = [0, 0]
        fx = self.camera_info_['fx']
        fy = self.camera_info_['fy']
        cx = self.camera_info_['cx']
        cy = self.camera_info_['cy']

        # Calculate 2D pixel coordinates from 3D positions (XYZ)
        if z_cam != 0:
            u = int(fx * x_cam / z_cam + cx)
            v = int(fy * y_cam / z_cam + cy)
            pixel = [u, v]
        return pixel

    def project_3d_covariance_to_2d(self, x_cam, y_cam, z_cam, cov_x, cov_y, cov_z):
        """
        Projects 3D covariances onto 2D covariances.
        """
        fx = self.camera_info_['fx']
        fy = self.camera_info_['fy']

        J = np.array([[fx / z_cam, 0, -fx * x_cam / z_cam**2],
                      [0, fy / z_cam, -fy * y_cam / z_cam**2]])

        covariance_3d = np.diag([cov_x, cov_y, cov_z])
        covariance_2d = J @ covariance_3d @ J.T
        return covariance_2d

    def depthToPoseMsg(self, pixel, depth):
        """
        Computes 3D projections of detections in the camera frame.
        """
        pose_msg = Pose()
        if self.camera_info_ is None:
            self.get_logger().warn("[Yolo2PoseNode::depthToPoseMsg] Camera intrinsic parameters are not available.")
            return pose_msg

        fx = self.camera_info_['fx']
        fy = self.camera_info_['fy']
        cx = self.camera_info_['cx']
        cy = self.camera_info_['cy']
        u = pixel[0]  # horizontal image coordinate
        v = pixel[1]  # vertical image coordinate
        d = depth  # depth

        x = d * (u - cx) / fx
        y = d * (v - cy) / fy

        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = float(d)
        pose_msg.orientation.w = 1.0

        return pose_msg

    def transform_pose(self, pose: Pose, tr: TransformStamped) -> Pose:
        """
        Converts 3D positions in the camera frame to the reference frame.
        """
        tf2_pose_msg = TF2Pose()
        tf2_pose_msg.position.x = pose.position.x
        tf2_pose_msg.position.y = pose.position.y
        tf2_pose_msg.position.z = pose.position.z
        tf2_pose_msg.orientation = pose.orientation

        try:
            transformed_pose = do_transform_pose(tf2_pose_msg, tr)

            # Check nor NaN values
            if np.isnan(transformed_pose.position.x) or np.isnan(transformed_pose.position.y) or np.isnan(transformed_pose.position.z):
                self.get_logger().error("[transform_pose] Transformed pose contains NaN in the position values)")
                return None
        except Exception as e:
            self.get_logger().error("[transform_pose] Error in transforming pose: {}".format(e))
            return None

        return transformed_pose

    def transform_pose_cov(self, pose: PoseWithCovarianceStamped, tr: TransformStamped) -> PoseWithCovarianceStamped:
        """
        Converts 3D pose with covariance from the frame in pose to the frame in tr.
        """
        try:
            pose_cov_stamped = do_transform_pose_with_covariance_stamped(pose, tr)
        except Exception as e:
            self.get_logger().error("[transform_pose_cov] Error in transforming pose with covariance: {}".format(e))
            return None

        return pose_cov_stamped

def main(args=None):
    rclpy.init(args=args)
    yolo2pose_node = Yolo2PoseNode()
    yolo2pose_node.get_logger().info("Yolo to Pose conversion node has started")
    rclpy.spin(yolo2pose_node)
    yolo2pose_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
