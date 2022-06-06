#!/usr/bin/env python3
import numpy as np
import cv2
from cv_bridge import CvBridge
import glob

from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory


class StereoCalibration(Node):
    def __init__(self):
        super().__init__("stereo_calibration")
        self.get_logger().info("Stereo calibration node is awake...")

        # Parameters declarations
        self.declare_parameter("board_dim", [6, 8])
        self.board_dim = (
            self.get_parameter("board_dim").get_parameter_value().integer_array_value
        )

        self.declare_parameter("width_image", 808)
        self.width_image = (
            self.get_parameter("width_image").get_parameter_value().integer_value
        )

        self.declare_parameter("height_image", 608)
        self.height_image = (
            self.get_parameter("height_image").get_parameter_value().integer_value
        )

        self.declare_parameter("square_size", "28.0")
        self.square_size = (
            self.get_parameter("square_size").get_parameter_value().double_value
        )

        self.declare_parameter("images_path", "auto")
        self.images_path = (
            self.get_parameter("images_path").get_parameter_value().string_value
        )

        if self.left_images_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.images_path = package_share_directory + "/calibration_images/left/"

        if self.right_images_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.images_path = package_share_directory + "/calibration_images/right/"

        self.declare_parameter("calibration_path", "auto")
        self.calibration_path = (
            self.get_parameter("calibration_path").get_parameter_value().string_value
        )

        if self.calibration_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.calibration_path = package_share_directory + "/calibration/"

        self.left_images = self.load_images(self.left_images_path + "/*.png")
        self.right_images = self.load_images(self.right_images_path + "/*.png")

    def load_images(self, path):
        filenames = glob.glob(path)
        filenames.sort()
        return [cv2.imread(img) for img in filenames]
