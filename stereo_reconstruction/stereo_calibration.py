#!/usr/bin/env python3

# Libraries
import sys 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
from time import sleep
import threading
from ament_index_python.packages import get_package_share_directory

class StereoCalibration():
    pass