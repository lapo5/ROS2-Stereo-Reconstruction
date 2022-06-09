#!/usr/bin/env python3

import sys
import numpy as np 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from rclpy.subscription import Subscription
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

from stereo_reconstruction.stereo_reconstruction import StereoReconstruction

from typing import List

import threading

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
            self.calibration_file = package_share_directory + "/calibration/calib_params_stereo.xml"
            
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
            
        self.stereo_reconstruction = StereoReconstruction(self.calibration_file)
        
        self.pcl_pub = self.create_publisher(PointCloud2, "test/pointcloud2", 1)

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
        img_left = self.current_frame_left
        img_right = self.current_frame_right
        
        disparity_map = self.stereo_reconstruction.disparity_from_stereovision(img_left, img_right)
        output_points, output_colors = self.stereo_reconstruction.pcl_from_disparity(disparity_map, img_left, img_right)
        
        pclmsg = PointCloud2()
        
        pclmsg.fields
        
        pclmsg.header.stamp = rclpy.Time.now()
        pclmsg.header.frame_id = "pointcloud"
        pclmsg.width = output_points.shape[1]
        pclmsg.height = output_points.shape[0]
        pclmsg.fields = [PointField('x',0,PointField.FLOAT32,1),
                                PointField('y',4,PointField.FLOAT32,1),
                                PointField('z',8,PointField.FLOAT32,1)]
        pclmsg.is_bigendian = False
        pclmsg.is_dense = False
        pclmsg.point_step = 12
        pclmsg.row_step = 12*output_points.shape[1]
        pclmsg.data = np.asarray(output_points,np.float32).tostring()

        self.pcl_pub.publish(pclmsg)




def main(args=None):

    rclpy.init(args=args)
    node = StereoReconstructionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Stereo calibration node] Node stopped cleanly")
        node.exit()
    except BaseException:
        node.get_logger().info("[Stereo calibration node] Exception:", file=sys.stderr)
        node.exit()
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
