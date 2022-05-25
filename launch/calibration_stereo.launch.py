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

    params = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_stereo_calibration.yaml')


    return LaunchDescription([
        Node(
            package='stereo_reconstruction',
            executable='stereo_calibration',
            name='stereo_calibration',
            parameters=[params],
        )
])