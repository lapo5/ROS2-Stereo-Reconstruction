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

def generate_launch_description():

    params = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params.yaml')


    params_allied_1 = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_cam1.yaml')
    params_allied_2 = os.path.join(get_package_share_directory("stereo_reconstruction"), 'params', 'params_cam2.yaml')

    node_recon = Node(
            package='stereo_reconstruction',
            executable='stereo_recon',
            name='stereo_recon',
            parameters=[params],
        )

    return LaunchDescription([
        #node_recon,

        Node(
            package='hal_allied_vision_camera',
            executable='av_node',
            name='av_node',
            parameters=[params_allied_1],
        ),
        
        Node(
            package='hal_allied_vision_camera',
            executable='av_node',
            name='av_node_2',
            parameters=[params_allied_2],
        ),
])
