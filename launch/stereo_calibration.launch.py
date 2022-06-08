import os

from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    params_stereo_calibration_filepath = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_stereo_calibration.yaml",
    )

    stereo_calibration_node = Node(
        package="stereo_reconstruction",
        executable="stereo_calibration",
        name="stereo_calibration",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[params_stereo_calibration_filepath],
    )

    return LaunchDescription([stereo_calibration_node])
