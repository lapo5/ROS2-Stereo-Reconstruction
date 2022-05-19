from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
import os
import sys
import yaml
from launch.substitutions import EnvironmentVariable
import pathlib
import launch.actions
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from launch.actions.execute_process import ExecuteProcess

def generate_launch_description():

    params = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_calibration_cam_left.yaml')
    params_av_node = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_cam_left_for_calibration.yaml')
    
    return LaunchDescription([
        Node(
            package='hal_allied_vision_camera',
            executable='calibration_node',
            name='calibration_node',
            output={
                    "stdout": "screen",
                    "stderr": "screen",
            },
            parameters=[params],
        ),

        Node(
            package='hal_allied_vision_camera',
            executable='av_node',
            name='av_node',
            output={
                    "stdout": "screen",
                    "stderr": "screen",
            },
            parameters=[params_av_node],
        )
])