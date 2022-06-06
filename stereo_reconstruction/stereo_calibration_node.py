import sys

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from typing import  List

from stereo_reconstruction.camera_calibration import camera_calibration



from stereo_reconstruction.stereo_acquisition_node import StereoAcquisition 


class StereoCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_calibration")
        self.get_logger().info("Calibration node is awake...")

        self.declare_parameter("acquisition_terminated", "False")

        # Parameters declarations
        
        self.declare_parameter("image_size", [800, 600])
        self.image_size: List[int] = (
            self.get_parameter("image_size").get_parameter_value().integer_array_value
        )
        
        self.get_logger().info(f"Image size: {self.image_size}")
        
        self.declare_parameter("chessboard_size", [6, 8])
        self.chessboard_size: List[int] = (
            self.get_parameter("chessboard_size").get_parameter_value().integer_array_value
        )
        
        self.get_logger().info(f"Chessboard size: {self.chessboard_size}")
        
        self.declare_parameter("square_size", 20.0)
        self.square_size: float = (
            self.get_parameter("square_size").get_parameter_value().double_value
        )
        
        self.get_logger().info(f"Chessboard size: {self.chessboard_size}")
                
        self.declare_parameter("images_path", "auto")
        self.images_path: str = (
            self.get_parameter("images_path").get_parameter_value().string_value
        )

        if self.images_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.images_path = package_share_directory + "/calibration_images/"

        self.left_images_path: str = self.images_path + "left/"
        self.right_images_path: str = self.images_path + "right/"
        
        self.get_logger().info(f"Dict camera_1: {camera_calibration.calibrate(self.left_images_path, self.chessboard_size, self.image_size)}")
        self.get_logger().info(f"Dict camera_2: {camera_calibration.calibrate(self.right_images_path, self.chessboard_size, self.image_size)}")
        
        
        
        
        

def main(args=None):

    rclpy.init(args=args)
    node = StereoCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Calibration node] Node stopped cleanly")
        node.exit()
    except BaseException:
        node.get_logger().info("[Calibration node] Exception:", file=sys.stderr)
        node.exit()
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    
