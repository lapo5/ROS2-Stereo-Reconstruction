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

from ament_index_python.packages import get_package_share_directory

# Class definition fo the estimator
class StereoReconstruction(Node):
    def __init__(self):
        super().__init__("stereo_recon")

        self.bridge = CvBridge()

        self.declare_parameter("publishers.disparity_map", "/disparity_map")
        self.disp_map_topic = self.get_parameter("publishers.disparity_map").value

        self.declare_parameter("subscribers.raw_frame_left", "/camera_left/raw_frame")
        self.raw_frame_left_topic = self.get_parameter("subscribers.raw_frame_left").value

        self.declare_parameter("subscribers.raw_frame_right", "/camera_right/raw_frame")
        self.raw_frame_right_topic = self.get_parameter("subscribers.raw_frame_right").value

        self.frame_right = None
        self.frame_left = None
        
        self.frame_left_sub = self.create_subscription(Image, self.raw_frame_left_topic, self.callback_frame_left, 1)

        self.frame_right_sub = self.create_subscription(Image, self.raw_frame_right_topic, self.callback_frame_right, 1)

        self.publisher_disp_map = self.create_publisher(Image, self.disp_map_topic, 1)

        self.declare_parameter("camera_module", "hal_allied_vision_camera")
        self.camera_module = self.get_parameter("camera_module").value

        package_share_directory = get_package_share_directory("stereo_reconstruction")
        self.calibration_camera_left_path = package_share_directory + "/calibration/calib_params_cam_left.json"
        self.calibration_camera_right_path = package_share_directory + "/calibration/calib_params_cam_right.json"

        with open(self.calibration_camera_left_path, "r") as readfile:
            self.cam_left_params = json.load(readfile)

        with open(self.calibration_camera_right_path, "r") as readfile:
            self.cam_right_params = json.load(readfile)

        self.K_left  = np.array(self.cam_left_params["mtx"], dtype=float).reshape(3, 3)
        self.D_left  = np.array(self.cam_left_params["dist"], dtype=float)
        self.D_left = self.D_left[0:4]
        self.K_right = np.array(self.cam_right_params["mtx"], dtype=float).reshape(3, 3)
        self.D_right = np.array(self.cam_right_params["dist"], dtype=float)
        self.D_right = self.D_right[0:4]

        (self.width, self.height) = (1000, 1000)

        self.declare_parameter("hz", "1.0")
        self.hz = float(self.get_parameter("hz").value)

        # Get the relative extrinsics between the left and right camera
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.zeros([3, 1], dtype=np.float32)
        # Baseline only on x axis: 6 cm
        self.T[0] = 0.06

        self.min_disp = 0
        # must be divisible by 16
        self.num_disp = 320
        self.max_disp = self.min_disp + self.num_disp

        self.block_size = 7
        self.stereo_good = cv2.StereoSGBM_create(minDisparity = self.min_disp,
                                        numDisparities = self.num_disp,
                                        blockSize = self.block_size,
                                        P1 = 8*self.block_size**2,
                                        P2 = 32*self.block_size**2,
                                        disp12MaxDiff = -1,
                                        uniquenessRatio = 10,
                                        speckleWindowSize = 0,
                                        speckleRange = 1)

        # We need to determine what focal length our undistorted images should have
        # in order to set up the camera matrices for initUndistortRectifyMap.  We
        # could use stereoRectify, but here we show how to derive these projection
        # matrices from the calibration and a desired height and field of view

        # We calculate the undistorted focal length:
        #
        #         h
        # -----------------
        #  \      |      /
        #    \    | f  /
        #     \   |   /
        #      \ fov /
        #        \|/
        self.stereo_fov_rad = 90 * (math.pi/180)  # 90 degree desired fov
        self.stereo_height_px = 1000          # 300x300 pixel stereo output
        self.stereo_focal_px = self.stereo_height_px/2 / math.tan(self.stereo_fov_rad/2)

        # We set the left rotation to identity and the right rotation
        # the rotation between the cameras
        self.R_left = np.eye(3)
        self.R_right = self.R

        # The stereo algorithm needs max_disp extra pixels in order to produce valid
        # disparity on the desired output region. This changes the width, but the
        # center of projection should be on the center of the cropped image
        self.stereo_width_px = self.stereo_height_px + self.max_disp
        self.stereo_size = (self.stereo_width_px, self.stereo_height_px)
        self.stereo_cx = (self.stereo_height_px - 1)/2 + self.max_disp
        self.stereo_cy = (self.stereo_height_px - 1)/2

        # Construct the left and right projection matrices, the only difference is
        # that the right projection matrix should have a shift along the x axis of
        # baseline*focal_length
        self.P_left = np.array([[self.stereo_focal_px,       0,              self.stereo_cx,  0],
                                [0,               self.stereo_focal_px, self.stereo_cy,  0],
                                [0,                     0,                  1,           0]])
        self.P_right = self.P_left.copy()
        self.P_right[0][3] = self.T[0]*self.stereo_focal_px

        # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
        # since we will crop the disparity later
        self.Q = np.array([[1, 0,       0,              -(self.stereo_cx - self.max_disp)],
                           [0, 1,       0,              -self.stereo_cy],
                           [0, 0,       0,              self.stereo_focal_px],
                           [0, 0,       -1/self.T[0],   0]])

        # Create an undistortion map for the left and right camera which applies the
        # rectification and undoes the camera distortion. This only has to be done
        # once
        self.m1type = cv2.CV_32FC1
        (self.lm1, self.lm2) = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, self.R_left, 
                                                                    self.P_left, self.stereo_size, self.m1type)

        (self.rm1, self.rm2) = cv2.fisheye.initUndistortRectifyMap(self.K_right, self.D_right, self.R_right, 
                                                                    self.P_right, self.stereo_size, self.m1type)
        self.undistort_rectify = {"left"  : (self.lm1, self.lm2),
                                  "right" : (self.rm1, self.rm2)}


        self.timer = self.create_timer(1.0/self.hz, self.estimate_3d_reconstr) # 50 Hz

        self.get_logger().info("[Stereo 3D Reconstruction] Node Ready")


    def undistort(self):
        # Undistort and crop the center of the frames
        self.center_undistorted = {"left" : cv2.remap(src = self.frame_left,
                                       map1 = self.undistort_rectify["left"][0],
                                       map2 = self.undistort_rectify["left"][1],
                                       interpolation = cv2.INTER_LINEAR),
                                    "right" : cv2.remap(src = self.frame_right,
                                       map1 = self.undistort_rectify["right"][0],
                                       map2 = self.undistort_rectify["right"][1],
                                       interpolation = cv2.INTER_LINEAR)}


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
        # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
        
        self.disparity = self.stereo_good.compute(self.center_undistorted["left"], 
                                self.center_undistorted["right"]).astype(np.float32) / 16.0

        # re-crop just the valid part of the disparity
        self.disparity = self.disparity[:,self.max_disp:]

        # convert disparity to 0-255 and color it
        self.disp_vis = 255*(self.disparity - self.min_disp)/ self.num_disp
        self.disp_color = cv2.applyColorMap(cv2.convertScaleAbs(self.disp_vis,1), cv2.COLORMAP_JET)
        self.color_image_left = cv2.cvtColor(self.center_undistorted["left"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

        self.color_image_right = cv2.cvtColor(self.center_undistorted["right"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

        self.disparity_map = self.color_image_left.copy()
        self.ind = self.disparity >= self.min_disp
        self.disparity_map[self.ind, 0] = self.disp_color[self.ind, 0]
        self.disparity_map[self.ind, 1] = self.disp_color[self.ind, 1]
        self.disparity_map[self.ind, 2] = self.disp_color[self.ind, 2]

        self.disparity = cv2.normalize(src=self.disparity, dst=self.disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        self.disparity = np.uint8(self.disparity)

        disp_msg =  self.bridge.cv2_to_imgmsg(self.disparity_map)
        disp_msg.header = Header()
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        disp_msg.header.frame_id = "camera_left_link"

        self.publisher_disp_map.publish(disp_msg)

        # points = cv2.reprojectImageTo3D(self.disparity, self.Q)
        # mask = self.disparity > self.disparity.min()
        # out_points = points[mask]

        # print(out_points)
        # print(out_points.shape)


    def estimate_3d_reconstr(self):

        if self.frame_right is not None and self.frame_left is not None:
            self.undistort()
            self.compute_disparity()


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