import os
import yaml

from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    params_allied_r_filepath = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_cam_right.yaml",
    )
    params_allied_l_filepath = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_cam_left.yaml",
    )

    params_stereo_filepath = os.path.join(
        get_package_share_directory("stereo_reconstruction"),
        "params",
        "params_stereo_reconstruction.yaml",
    )

    # with open(configFilepath, 'r') as file:
    #     configParams = yaml.safe_load(file)['stereo_acquisition_node']['ros__parameters']

    node_cam_right = Node(
        package="hal_allied_vision_camera",
        executable="av_node",
        name="av_node_right",
        parameters=[params_allied_r_filepath],
    )

    node_cam_left = Node(
        package="hal_allied_vision_camera",
        executable="av_node",
        name="av_node_left",
        parameters=[params_allied_l_filepath],
    )

    stereo_acquisition_node = Node(
        package="stereo_reconstruction",
        executable="stereo_reconstruction",
        name="stereo_reconstruction",
        parameters=[params_stereo_filepath],
    )
    
    static_transform_pcl = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '3.14', 'world', "pointcloud"]
        )

    return LaunchDescription([node_cam_right, node_cam_left, stereo_acquisition_node, static_transform_pcl])
