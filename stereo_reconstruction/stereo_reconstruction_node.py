#!/usr/bin/env python3

from email.header import Header
import sys
import numpy as np
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from rclpy.subscription import Subscription
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

from stereo_reconstruction.stereo_reconstruction import StereoReconstruction

from typing import List

import threading

import cv2


class StereoReconstructionNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_reconstruction")
        self.get_logger().info("Stereo reconstruction node is awake...")

        # Parameters declarations
        self.declare_parameter("calibration_file", "auto")
        self.calibration_file = (
            self.get_parameter("calibration_file").get_parameter_value().string_value
        )

        if self.calibration_file == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.calibration_file = (
                package_share_directory + "/calibration/calib_params_stereo.xml"
            )

        self.get_logger().info(f"calibration_file: {self.calibration_file}")

        self.declare_parameter("subscribers.camera_left", "/camera_left/raw_frame")
        self.camera_left_topic: str = (
            self.get_parameter("subscribers.camera_left")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("subscribers.camera_right", "/camera_right/raw_frame")
        self.camera_right_topic: str = (
            self.get_parameter("subscribers.camera_right")
            .get_parameter_value()
            .string_value
        )

        self.bridge = CvBridge()
        self.current_frame_left = []
        self.current_frame_right = []

        reconstruction_parameters = {
            "numDisparities": 5 * 16,
            "blockSize": 4 * 2 + 5,
            "preFilterType": 0,
            "preFilterSize": 6 * 2 + 5,
            "preFilterCap": 16,
            "textureThreshold": 12,
            "uniquenessRatio": 12,
            "speckleRange": 13,
            "speckleWindowSize": 5 * 2,
            "disp12MaxDiff": 3,
            "minDisparity": 4,
        }

        self.stereo_reconstruction = StereoReconstruction(
            self.calibration_file, reconstruction_parameters
        )

        self.pcl_pub = self.create_publisher(PointCloud2, "pointcloud2", 1)
        self.disparity_pub = self.create_publisher(Image, "disparity_map", 1)

        self.frame_left_sub: Subscription = self.create_subscription(
            Image, self.camera_left_topic, self.callback_frame_left, 1
        )

        self.frame_right_sub: Subscription = self.create_subscription(
            Image, self.camera_right_topic, self.callback_frame_right, 1
        )

        self.generate_pcl_thread = threading.Thread(
            target=self.generate_pcl_thread, daemon=True
        )
        self.generate_pcl_thread.start()

    def callback_frame_left(self, msg):
        try:
            self.current_frame_left = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except CvBridgeError as e:
            print(e)

    def callback_frame_right(self, msg):
        try:
            self.current_frame_right = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except CvBridgeError as e:
            print(e)

    def generate_pcl_thread(self):

        while len(self.current_frame_left) == 0 or len(self.current_frame_right) == 0:
            self.get_logger().warn("Waiting for both frame acquisition to start...")
            time.sleep(1)

        while True:
            img_left = self.current_frame_left
            img_right = self.current_frame_right

            input_image_height, input_image_width, input_image_channels = img_left.shape
            (
                input_image_height,
                input_image_width,
                input_image_channels,
            ) = img_right.shape

            disparity_map = self.stereo_reconstruction.disparity_from_stereovision(
                img_left, img_right
            )
            (
                output_points,
                output_colors,
            ) = self.stereo_reconstruction.pcl_from_disparity(
                disparity_map, img_left, img_right
            )

            xyz = np.array(np.hstack([output_points, output_colors]), dtype=np.float32)
            N = len(output_points)

            pclmsg = PointCloud2()

            pclmsg.header.stamp = self.get_clock().now().to_msg()
            pclmsg.header.frame_id = "pointcloud"
            pclmsg.height = 1
            pclmsg.width = N

            pclmsg.fields = [
                PointField(name="x", count=1, datatype=PointField.FLOAT32, offset=0),
                PointField(name="y", count=1, datatype=PointField.FLOAT32, offset=4),
                PointField(name="z", count=1, datatype=PointField.FLOAT32, offset=8),
                PointField(name="r", count=1, datatype=PointField.FLOAT32, offset=12),
                PointField(name="g", count=1, datatype=PointField.FLOAT32, offset=16),
                PointField(name="b", count=1, datatype=PointField.FLOAT32, offset=20),
            ]
            pclmsg.is_bigendian = False
            pclmsg.is_dense = True
            pclmsg.point_step = 24

            pclmsg.row_step = N * pclmsg.point_step
            pclmsg.data = xyz.tostring()

            self.pcl_pub.publish(pclmsg)

            disparity = self.bridge.cv2_to_imgmsg(disparity_map)
            disparity.header = pclmsg.header

            self.disparity_pub.publish(disparity)

    def nothing(self, x):
        pass

    def __generate_pcl_thread_test_to_calibrate(self):

        while len(self.current_frame_left) == 0 or len(self.current_frame_right) == 0:
            self.get_logger().warn("Waiting for both frame acquisition to start...")
            time.sleep(1)

        ##################### TEST #####################

        (
            input_image_height,
            input_image_width,
            input_image_channels,
        ) = self.current_frame_left.shape

        cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disp", input_image_height, input_image_width)

        cv2.createTrackbar("numDisparities", "disp", 1, 17, self.nothing)
        cv2.createTrackbar("blockSize", "disp", 5, 50, self.nothing)
        cv2.createTrackbar("preFilterType", "disp", 1, 1, self.nothing)
        cv2.createTrackbar("preFilterSize", "disp", 2, 25, self.nothing)
        cv2.createTrackbar("preFilterCap", "disp", 5, 62, self.nothing)
        cv2.createTrackbar("textureThreshold", "disp", 10, 100, self.nothing)
        cv2.createTrackbar("uniquenessRatio", "disp", 15, 100, self.nothing)
        cv2.createTrackbar("speckleRange", "disp", 0, 100, self.nothing)
        cv2.createTrackbar("speckleWindowSize", "disp", 3, 25, self.nothing)
        cv2.createTrackbar("disp12MaxDiff", "disp", 5, 25, self.nothing)
        cv2.createTrackbar("minDisparity", "disp", 5, 25, self.nothing)

        while True:
            img_left = self.current_frame_left
            img_right = self.current_frame_right

            imgR_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            imgL_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

            # Applying stereo image rectification on the left image
            Left_nice = cv2.remap(
                imgL_gray,
                self.stereoMapL_x,
                self.stereoMapL_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0,
            )

            # Applying stereo image rectification on the right image
            Right_nice = cv2.remap(
                imgR_gray,
                self.stereoMapR_x,
                self.stereoMapR_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0,
            )

            # Updating the parameters based on the trackbar positions
            numDisparities = cv2.getTrackbarPos("numDisparities", "disp") * 16
            blockSize = cv2.getTrackbarPos("blockSize", "disp") * 2 + 5
            preFilterType = cv2.getTrackbarPos("preFilterType", "disp")
            preFilterSize = cv2.getTrackbarPos("preFilterSize", "disp") * 2 + 5
            preFilterCap = cv2.getTrackbarPos("preFilterCap", "disp")
            textureThreshold = cv2.getTrackbarPos("textureThreshold", "disp")
            uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disp")
            speckleRange = cv2.getTrackbarPos("speckleRange", "disp")
            speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disp") * 2
            disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disp")
            minDisparity = cv2.getTrackbarPos("minDisparity", "disp")

            """
            - stereo.setNumDisparities: increasing the number of disparities increases
                the accuracy of the disparity map 
                
            - self.stereo.setBlockSize: size of the sliding window used for block matching 
                to find corresponding pixels in a rectified stereo image pair. Higher means 
                more smooth disparity.
                
            - self.stereo.setPreFilterType: CV_STEREO_BM_XSOBEL or CV_STEREO_BM_NORMALIZED_RESPONSE
            
            - self.stereo.setPreFilterSize: Window size of the filter
            
            - self.stereo.setPreFilterCap: Limits the filtered output to a specific value.
            
            - self.stereo.setTextureThreshold: Filters out areas that do not have enough texture 
                information for reliable matching.

            - self.stereo.setUniquenessRatio: Another post-filtering step. If the best matching 
                disparity is not sufficiently better than every other disparity in the search 
                range, the pixel is filtered out. 
                
            - self.stereo.setSpeckleRange and self.stereo.setSpeckleWindowSize: 
                Speckles are produced near the boundaries of the objects, where the matching 
                window catches the foreground on one side and the background on the other. 
                To get rid of these artifacts we apply speckle filter which has two parameters. 
                The speckle range defines how close the disparity values should be to be considered 
                as part of the same blob. The speckle window size is the number of pixels below 
                which a disparity blob is dismissed as “speckle”.

            - self.stereo.setDisp12MaxDiff: Pixels are matched both ways, from the left image 
                to the right image and from the right image to left image. disp12MaxDiff 
                defines the maximum allowable difference between the original left pixel 
                and the back-matched pixel.

            - self.stereo.setMinDisparity: 
            """
            # Setting the updated parameters before computing disparity map
            self.stereo.setNumDisparities(numDisparities)
            self.stereo.setBlockSize(blockSize)
            self.stereo.setPreFilterType(preFilterType)
            self.stereo.setPreFilterSize(preFilterSize)
            self.stereo.setPreFilterCap(preFilterCap)
            self.stereo.setTextureThreshold(textureThreshold)
            self.stereo.setUniquenessRatio(uniquenessRatio)
            self.stereo.setSpeckleRange(speckleRange)
            self.stereo.setSpeckleWindowSize(speckleWindowSize)
            self.stereo.setDisp12MaxDiff(disp12MaxDiff)
            self.stereo.setMinDisparity(minDisparity)

            disparity = self.stereo.compute(Left_nice, Right_nice)

            disparity = disparity.astype(np.float32)

            disparity = (disparity / 16.0 - minDisparity) / numDisparities

            cv2.imshow("disp", disparity)
            cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)
    node = StereoReconstructionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Stereo calibration node] Node stopped cleanly")
        # node.exit()
    except BaseException:
        node.get_logger().info("[Stereo calibration node] Exception:", file=sys.stderr)
        # node.exit()
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
