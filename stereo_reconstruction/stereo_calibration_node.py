import sys

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from typing import List

from stereo_reconstruction.camera_calibration import CameraCalibration
from stereo_reconstruction.stereo_calibration import StereoCalibration


class StereoCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_calibration")
        self.get_logger().info("Calibration node is awake...")

        self.declare_parameter("acquisition_terminated", "False")

        # Parameters declarations

        self.declare_parameter("image_size", [800, 600])
        self.image_size = (
            self.get_parameter("image_size").get_parameter_value().integer_array_value
        )

        self.declare_parameter("chessboard_size", [6, 8])
        self.chessboard_size = (
            self.get_parameter("chessboard_size")
            .get_parameter_value()
            .integer_array_value
        )
        
        self.declare_parameter("square_size", 20.0)
        self.square_size = (
            self.get_parameter("square_size").get_parameter_value().double_value
        )

        self.declare_parameter("minimum_valid_images", 20)
        self.minimum_valid_images = (
            self.get_parameter("minimum_valid_images")
            .get_parameter_value()
            .integer_value
        )


        self.declare_parameter("calibration_path", "auto")
        self.calibration_path = (
            self.get_parameter("calibration_path").get_parameter_value().string_value
        )

        if self.calibration_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.calibration_path = package_share_directory + "/calibration/"

        self.declare_parameter("images_path", "auto")
        self.images_path = (
            self.get_parameter("images_path").get_parameter_value().string_value
        )

        if self.images_path == "auto":
            package_share_directory = get_package_share_directory(
                "stereo_reconstruction"
            )
            self.images_path = package_share_directory + "/calibration_images/"

        self.left_images_path = self.images_path + "left/"
        self.right_images_path = self.images_path + "right/"

        #################### SINGLE CAMERA CALIBRATION ####################

        cam_l_calibration = CameraCalibration()
        cam_r_calibration = CameraCalibration()

        left_cam_params = cam_l_calibration.calibrate(
            self.left_images_path,
            self.chessboard_size,
            self.image_size,
            self.minimum_valid_images,
        )
        right_cam_params = cam_r_calibration.calibrate(
            self.right_images_path,
            self.chessboard_size,
            self.image_size,
            self.minimum_valid_images,
        )

        if not left_cam_params or not right_cam_params:
            raise Exception(
                "Minimum number of valid images not reached in camera calibration."
            )

        try:
            cam_l_calibration.save_params(
                self.calibration_path, "calib_params_left_cam.json"
            )
            self.get_logger().info(
                f'Left camera calibrated. Mean error: {left_cam_params["mean_error"]:.5f}.'
            )
            cam_r_calibration.save_params(
                self.calibration_path, "calib_params_right_cam.json"
            )
            self.get_logger().info(
                f'Right camera calibrated. Mean error: {right_cam_params["mean_error"]:.5f}.'
            )
        except Exception as e:
            self.get_logger().info(f"Exception: {e}")

        #################### STEREO CALIBRATION ####################

        stereo_calibration = StereoCalibration()

        try:
            stereo_params = stereo_calibration.calibrate(
                left_cam_params, right_cam_params
            )
            self.get_logger().info(
                f'Stereo system calibrated. Rectification roi_right: {stereo_params["roi1"]} and roi_left:{stereo_params["roi2"]}.'
            )
            self.get_logger().info(f'Translation: {stereo_params["T"]}.')
        except Exception as e:
            self.get_logger().info(f"Exception: {e}")

        stereo_calibration.save_params(
            self.calibration_path, "calib_params_stereo.json"
        )


def main(args=None):

    rclpy.init(args=args)
    node = StereoCalibrationNode()
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
