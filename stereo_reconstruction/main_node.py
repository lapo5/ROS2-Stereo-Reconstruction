import sys
import rclpy

from stereo_reconstruction.stereo_acquisition_node import StereoAcquisition 
from stereo_reconstruction.camera_calibration import camera_calibration


def main(args=None):

    rclpy.init(args=args)

    acquisition_node = StereoAcquisition()

    try:
        rclpy.spin(acquisition_node)
    except KeyboardInterrupt:
        acquisition_node.get_logger().info("Acquistion node stopped cleanly")
    except BaseException:
        acquisition_node.get_logger().info(
            "Exception in Acquistion node:", file=sys.stderr
        )
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
