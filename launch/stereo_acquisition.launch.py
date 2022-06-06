import os

from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    params_allied_left = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_cam_left.yaml')
    params_allied_right = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_cam_right.yaml')

    params = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_stereo_calibration.yaml",
    )

    node_cam_right = Node(
        package="hal_allied_vision_camera",
        executable="av_node",
        name="av_node_right",
        parameters=[params_allied_right],
    )

    node_cam_left = Node(
        package="hal_allied_vision_camera",
        executable="av_node",
        name="av_node_left",
        parameters=[params_allied_left],
    )
    
    stereo_acquisition_node = Node(
        package="stereo_reconstruction",
        executable="stereo_acquisition_node",
        name="stereo_acquisition_node",
        parameters=[params],
    ) 

    return LaunchDescription([node_cam_right, node_cam_left, stereo_acquisition_node])
