import os

from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    params = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_stereo_calibration.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="stereo_reconstruction",
                executable="main_node",
                name="main_node",
                parameters=[params],
            )
        ]
    )
