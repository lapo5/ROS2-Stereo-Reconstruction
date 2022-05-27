#!/usr/bin/env python3

# Libraries
import sys
import json
import numpy as np
import cv2
import math
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from functools import partial

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

import tf2_ros
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

from ament_index_python.packages import get_package_share_directory

# Class definition fo the estimator
class StereoReconstruction(Node):
    def __init__(self):
        super().__init__("stereo_recon")

        self.bridge = CvBridge()

        self.declare_parameter("publishers.disparity_map", "/disparity_map")
        self.disp_map_topic = self.get_parameter("publishers.disparity_map").value

        self.declare_parameter("publishers.pointcloud", "/pointcloud")
        self.pointcloud_topic = self.get_parameter("publishers.pointcloud").value

        self.declare_parameter("subscribers.raw_frame_left", "/camera_left/raw_frame")
        self.raw_frame_left_topic = self.get_parameter("subscribers.raw_frame_left").value

        self.declare_parameter("subscribers.raw_frame_right", "/camera_right/raw_frame")
        self.raw_frame_right_topic = self.get_parameter("subscribers.raw_frame_right").value

        self.declare_parameter("frames.stereo_link", "camera_left_link")
        self.stereo_link = self.get_parameter("frames.stereo_link").value

        self.declare_parameter("enable_compute_pointcloud", False)
        self.enable_compute_pointcloud = self.get_parameter("enable_compute_pointcloud").value

        self.frame_right = None
        self.frame_left = None
        
        self.frame_left_sub = self.create_subscription(Image, self.raw_frame_left_topic, self.callback_frame_left, 1)

        self.frame_right_sub = self.create_subscription(Image, self.raw_frame_right_topic, self.callback_frame_right, 1)

        self.publisher_disp_map = self.create_publisher(Image, self.disp_map_topic, 1)

        self.publisher_left_image_rect = self.create_publisher(Image, "/left_image_rect", 1)
        self.publisher_right_image_rect = self.create_publisher(Image, "/right_image_rect", 1)

        self.publisher_pointcloud = self.create_publisher(PointCloud2, self.pointcloud_topic, 1)

        package_share_directory = get_package_share_directory("stereo_reconstruction")
        self.calibration_camera_left_path = package_share_directory + "/calibration/calib_params_cam_left.json"
        self.calibration_camera_right_path = package_share_directory + "/calibration/calib_params_cam_right.json"
        self.calibration_stereo_path = package_share_directory + "/calibration/stereo_calib_params.json"

        with open(self.calibration_camera_left_path, "r") as readfile:
            self.cam_left_params = json.load(readfile)

        with open(self.calibration_camera_right_path, "r") as readfile:
            self.cam_right_params = json.load(readfile)

        with open(self.calibration_stereo_path, "r") as readfile:
            self.stereo_params = json.load(readfile)

        self.K_left  = np.array(self.cam_left_params["mtx"], dtype=float).reshape(3, 3)
        self.D_left  = np.array(self.cam_left_params["dist"], dtype=float)
        
        self.K_right = np.array(self.cam_right_params["mtx"], dtype=float).reshape(3, 3)
        self.D_right = np.array(self.cam_right_params["dist"], dtype=float)

        self.stereo_R   = np.array(self.stereo_params["R"], dtype=float).reshape(3, 3)
        self.stereo_T   = np.array(self.stereo_params["T"], dtype=float)

        self.stereo_R1  = np.array(self.stereo_params["R1"], dtype=float).reshape(3, 3)
        self.stereo_R2  = np.array(self.stereo_params["R2"], dtype=float).reshape(3, 3)

        self.stereo_P1  = np.array(self.stereo_params["P1"], dtype=float).reshape(3, 4)
        self.stereo_P2  = np.array(self.stereo_params["P2"], dtype=float).reshape(3, 4)

        self.stereo_Q  = np.array(self.stereo_params["Q"], dtype=float).reshape(4, 4)

        self.get_logger().info("self.stereo_Q: {0}".format(self.stereo_Q))

        self.declare_parameter("image_width", 1000)
        self.width = self.get_parameter("image_width").value

        self.declare_parameter("image_height", 1000)
        self.height = self.get_parameter("image_height").value

        self.declare_parameter("hz", "0.5")
        self.hz = float(self.get_parameter("hz").value)

        # Get the relative extrinsics between the left and right camera
        self.R = self.stereo_R
        self.T = self.stereo_T

        self.min_disp = 16
        # must be divisible by 16
        self.num_disp = 112 - self.min_disp

        self.block_size = 7
        self.stereo = cv2.StereoSGBM_create(minDisparity = self.min_disp,
                                        numDisparities = self.num_disp,
                                        blockSize = self.block_size,
                                        P1 = 8*self.block_size**2,
                                        P2 = 32*self.block_size**2,
                                        disp12MaxDiff = -1,
                                        uniquenessRatio = 10,
                                        speckleWindowSize = 0,
                                        speckleRange = 1)

        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, self.stereo_R1, self.stereo_P1, (self.width, self.height), cv2.CV_32FC1)

        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, self.stereo_R2, self.stereo_P2, (self.width, self.height), cv2.CV_32FC1)

        self.timer = self.create_timer(1.0/self.hz, self.estimate_3d_reconstr)

        self.get_logger().info("[Stereo 3D Reconstruction] Node Ready")


    def callback_frame_left(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        if len(frame.shape) == 3:
            self.frame_left = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.frame_left = frame


    def callback_frame_right(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        if len(frame.shape) == 3:
            self.frame_right = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.frame_right = frame


    def compute_disparity(self):
        
        self.get_logger().info("[Stereo 3D Reconstruction] Computing Disparity")
        frame_left = self.frame_left.copy()
        frame_right = self.frame_right.copy()

        left_rectified = cv2.remap(frame_left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        right_rectified = cv2.remap(frame_right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        self.disparity = self.stereo.compute(left_rectified, 
                                right_rectified)

        self.disparity = self.disparity.astype(np.float32)
        self.disparity_map = 255*(self.disparity - self.min_disp)/ self.num_disp
        cv2.normalize(src=self.disparity_map, dst=self.disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        self.disparity_map = np.uint8(self.disparity_map)

        disp_msg =  self.bridge.cv2_to_imgmsg(self.disparity_map)
        disp_msg.header = Header()
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        disp_msg.header.frame_id = self.stereo_link
        self.publisher_disp_map.publish(disp_msg)

        left_image_rect_msg =  self.bridge.cv2_to_imgmsg(left_rectified)
        left_image_rect_msg.header = Header()
        left_image_rect_msg.header.stamp = self.get_clock().now().to_msg()
        left_image_rect_msg.header.frame_id = self.stereo_link
        self.publisher_left_image_rect.publish(left_image_rect_msg)

        right_image_rect_msg =  self.bridge.cv2_to_imgmsg(right_rectified)
        right_image_rect_msg.header = Header()
        right_image_rect_msg.header.stamp = self.get_clock().now().to_msg()
        right_image_rect_msg.header.frame_id = self.stereo_link
        self.publisher_right_image_rect.publish(right_image_rect_msg)


    def compute_pointcloud(self):
        
        self.get_logger().info("[Stereo 3D Reconstruction] Computing Pointcloud")
        points = cv2.reprojectImageTo3D(self.disparity, self.stereo_Q)
        
        # Remove INF values from point cloud
        points[points == float('+inf')] = 0
        points[points == float('-inf')] = 0
        
        # Get rid of points with value 0 (i.e no depth)
        mask_map = self.disparity > self.min_disp

        # Mask colors and points
        out_points = points[mask_map]

        self.get_logger().info("PC: {0}".format(out_points))
        self.get_logger().info("PC shape: {0}".format(out_points.shape))

        pc_msg =  PointCloud2()
        pc_msg.header = Header()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = self.stereo_link

        x_field = PointField()
        x_field.name = 'x'
        x_field.offset = 0
        x_field.datatype = PointField.FLOAT32
        x_field.count = 1

        y_field = PointField()
        y_field.name = 'y'
        y_field.offset = 4
        y_field.datatype = PointField.FLOAT32
        y_field.count = 1

        z_field = PointField()
        z_field.name = 'z'
        z_field.offset = 8
        z_field.datatype = PointField.FLOAT32
        z_field.count = 1

        pc_msg.fields = [x_field, y_field, z_field]

        pc_msg.height = out_points.shape[0]
        pc_msg.width = out_points.shape[1]

        pc_msg.is_bigendian = False

        pc_msg.point_step = 4
        pc_msg.row_step = 12

        pc_msg.is_dense = True

        data = np.asarray(out_points, np.float32).tostring()
        pc_msg.data = data

        self.publisher_pointcloud.publish(pc_msg)
        

    def estimate_3d_reconstr(self):

        if self.frame_right is not None and self.frame_left is not None:
            self.compute_disparity()
            if self.enable_compute_pointcloud:
                self.compute_pointcloud()


def main(args=None):
    rclpy.init(args=args)
    node = StereoReconstruction()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[Stereo 3D Reconstruction] stopped cleanly')
    except BaseException:
        node.get_logger().info('[Stereo 3D Reconstruction] Exception:', file=sys.stderr)
        raise
    finally:
        rclpy.shutdown() 


# Main
if __name__ == '__main__':
    main()