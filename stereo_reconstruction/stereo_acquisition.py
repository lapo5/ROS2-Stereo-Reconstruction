#!/usr/bin/env python3

# Libraries
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
from time import sleep
import threading
from ament_index_python.packages import get_package_share_directory


class StereoAcquisition(Node):
    def __init__(self):
        super().__init__("stereo_acquisition")
        self.get_logger().info("Acquisition node is awake...")

        # Parameters declarations
        self.declare_parameter("number_of_images_to_save", 10)
        self.number_of_images_to_save = (
            self.get_parameter("number_of_images_to_save")
            .get_parameter_value()
            .integer_value
        )

        self.declare_parameter("minimum_valid_images", 10)
        self.min_valid_images = (
            self.get_parameter("minimum_valid_images")
            .get_parameter_value()
            .integer_value
        )

        self.declare_parameter("subscribers.camera_left", "/camera_left/raw_frame")
        self.camera_left_topic = (
            self.get_parameter("subscribers.camera_left")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("subscribers.camera_right", "/camera_right/raw_frame")
        self.camera_right_topic = (
            self.get_parameter("subscribers.camera_right")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("auto_capture.mode", "True")
        self.auto_capture = (
            self.get_parameter("auto_capture.mode").get_parameter_value().bool_value
        )

        self.declare_parameter("auto_capture.time_for_frame", True)
        self.time_for_frame = (
            self.get_parameter("auto_capture.time_for_frame")
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter("images_path", "auto")
        self.images_path = (
            self.get_parameter("images_path").get_parameter_value().string_value
        )

        if self.images_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.images_path = package_share_directory + "/calibration_images/"

        self.bridge = CvBridge()
        self.counter_left_images = 0
        self.counter_right_images = 0

        self.current_frame_left = []
        self.current_frame_right = []

        self.frame_left_sub = self.create_subscription(
            Image, self.camera_left_topic, self.callback_frame_left, 1
        )

        self.frame_right_sub = self.create_subscription(
            Image, self.camera_right_topic, self.callback_frame_right, 1
        )

        self.aquisition_thread = threading.Thread(
            target=self.acquisition_process, daemon=True
        )
        self.aquisition_thread.start()

    def callback_frame_left(self, msg):
        self.current_frame_left = self.bridge.imgmsg_to_cv2(msg)

    def callback_frame_right(self, msg):
        self.current_frame_right = self.bridge.imgmsg_to_cv2(msg)

    def acquisition_process(self):
        if self.auto_capture:
            self.get_logger().info(
                "\nAuto-Capture - Taking 1 Frame every {0} secs \n\n".format(
                    self.time_for_frame
                )
            )
            self.timer = self.create_timer(self.time_for_frame, self.save_frame)
        else:
            self.get_logger().info(
                "\n======== KEYBOARD COMMANDS ========\n\nq - quit pictures acquisition\nc - capture actual frame\n\n"
            )

        while (
            self.counter_left_images <= self.number_of_images_to_save
            and self.counter_right_images <= self.number_of_images_to_save
        ):

            horizontal_concat = np.concatenate(
                (self.current_frame_left, self.current_frame_right), axis=1
            )
            cv2.imshow("LiveCamera -- LEFT & RIGHT", horizontal_concat)
            key = cv2.waitKey(0)

            if key == ord("q"):
                self.get_logger().info("Calibration process has been stopped.")
                break
            elif key == ord("c"):
                self.save_frame()

        self.get_logger().info(
            f"Left images saved: {self.counter_left_images} right images saved: {self.counter_right_images}.\n"
        )
        cv2.destroyAllWindows()
        self.stop_acquisition = True

    def save_frame(self):

        left_img = self.current_frame_left
        right_img = self.current_frame_right

        cv2.imwrite(
            self.images_path + "left/image_" + int(self.counter_left_images) + ".png",
            left_img,
        )
        cv2.imwrite(
            self.images_path + "right/image_" + int(self.counter_right_images) + ".png",
            right_img,
        )

        self.counter_left_images += 1
        self.counter_right_images += 1
